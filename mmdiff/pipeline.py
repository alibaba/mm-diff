import os
import copy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch

from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin, text_encoder_attn_modules, text_encoder_mlp_modules, PatchedLoraProjection
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from .attention_processor import LoRASelfAttnProcessor, LoRACrossAttnProcessor
from .model import replace_text_encoder_forward, CLIPImageEncoder, CLIPEmbedsProjHead, FaceEmbedsProjHead, MLP


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class MMDiffStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    def load_from_checkpoint(
        self,
        image_encoder_path,
        mmdiff_ckpt,
        device="cuda",
        fuse_lora=False,
    ):
        self.image_encoder = CLIPImageEncoder.from_pretrained(image_encoder_path, local_files_only=True).to(device=device, dtype=torch.float16)

        self.obj_embeds_proj = CLIPEmbedsProjHead(
            in_dim=self.image_encoder.visual_projection.out_features,
            hidden_dim=self.image_encoder.visual_projection.out_features,
            out_dim_1=self.text_encoder.get_input_embeddings().weight.shape[-1],
            out_dim_2=self.text_encoder_2.get_input_embeddings().weight.shape[-1],
        ).to(device=device, dtype=torch.float16)

        self.face_embeds_proj = FaceEmbedsProjHead(
            face_embeds_dim=512,
            clip_embeds_dim=self.image_encoder.vision_model.config.hidden_size,
            corss_attention_dim=self.unet.config.cross_attention_dim,
            num_tokens=4,
        ).to(device=device, dtype=torch.float16)

        self.mm_fuse_one = MLP(
            in_dim=self.text_encoder.get_input_embeddings().weight.shape[-1] * 2,
            out_dim=self.text_encoder.get_input_embeddings().weight.shape[-1],
            hidden_dim=self.text_encoder.get_input_embeddings().weight.shape[-1],
            use_residual=False,
        ).to(device=device, dtype=torch.float16)

        self.mm_fuse_two = MLP(
            in_dim=self.text_encoder_2.get_input_embeddings().weight.shape[-1] * 2,
            out_dim=self.text_encoder_2.get_input_embeddings().weight.shape[-1],
            hidden_dim=self.text_encoder_2.get_input_embeddings().weight.shape[-1],
            use_residual=False,
        ).to(device=device, dtype=torch.float16)

        self.object_infos = {"object_embeds": None, "image_token_mask": None}
        replace_text_encoder_forward(self.text_encoder, self.mm_fuse_one, self.object_infos)

        self.object_infos_2 = {"object_embeds": None, "image_token_mask": None}
        replace_text_encoder_forward(self.text_encoder_2, self.mm_fuse_two, self.object_infos_2)

        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                attn_procs[name] = LoRASelfAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=64,
                    fuse_lora=fuse_lora,
                ).to(device, dtype=torch.float16)
            else:
                attn_procs[name] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=64,
                    fuse_scale=1.0,
                    num_image_tokens=4,
                    fuse_lora=fuse_lora,
                ).to(device, dtype=torch.float16)
        self.unet.set_attn_processor(attn_procs)
        unet_adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())

        unet_adapter_ckpt_name = f"unet_adapter_weights.bin"
        unet_adapter_state_dict = torch.load(os.path.join(mmdiff_ckpt, unet_adapter_ckpt_name), map_location=device)

        unet_adapter_modules.load_state_dict(unet_adapter_state_dict, strict=True)

        print(f"-- Load UNet adapter weights from {os.path.join(mmdiff_ckpt, unet_adapter_ckpt_name)}")

        text_encoder_adapter_ckpt_name = f"text_encoder_adapter_weights.bin"
        text_encoder_adapter_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(mmdiff_ckpt, unet_config=self.unet.config, weight_name=text_encoder_adapter_ckpt_name)

        text_encoder_state_dict = {k: v for k, v in text_encoder_adapter_state_dict.items() if "text_encoder." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(text_encoder_state_dict, network_alphas=network_alphas, text_encoder=self.text_encoder)

        text_encoder_state_dict = {k: v for k, v in text_encoder_adapter_state_dict.items() if "text_encoder_2." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(text_encoder_state_dict, network_alphas=network_alphas, text_encoder=self.text_encoder_2)

        print(f"-- Load Text encoder adapter weights from {os.path.join(mmdiff_ckpt, text_encoder_adapter_ckpt_name)}")

        custom_ckpt_name = f"custom_weights.bin"
        custom_state_dict = torch.load(os.path.join(mmdiff_ckpt, custom_ckpt_name), map_location=device)
        self.image_encoder_bak = copy.deepcopy(self.image_encoder)
        self.image_encoder.load_state_dict(custom_state_dict["image_encoder"], strict=False)
        self.obj_embeds_proj.load_state_dict(custom_state_dict["obj_embeds_proj"], strict=True)
        self.face_embeds_proj.load_state_dict(custom_state_dict["face_embeds_proj"], strict=True)
        self.mm_fuse_one.load_state_dict(custom_state_dict["mm_fuse_one"], strict=True)
        self.mm_fuse_two.load_state_dict(custom_state_dict["mm_fuse_two"], strict=True)

        print(f"-- Load custom weights from {os.path.join(mmdiff_ckpt, custom_ckpt_name)}")

        if fuse_lora:
            self.custom_fuse_lora()
    
    def custom_fuse_lora(self, fuse_text_encoder=True, fuse_unet=True, lora_scale=1.0):
        def fuse_text_encoder_lora(text_encoder):
            for _, attn_module in text_encoder_attn_modules(text_encoder):
                if isinstance(attn_module.q_proj, PatchedLoraProjection):
                    attn_module.q_proj._fuse_lora(lora_scale)
                    attn_module.k_proj._fuse_lora(lora_scale)
                    attn_module.v_proj._fuse_lora(lora_scale)
                    attn_module.out_proj._fuse_lora(lora_scale)

            for _, mlp_module in text_encoder_mlp_modules(text_encoder):
                if isinstance(mlp_module.fc1, PatchedLoraProjection):
                    mlp_module.fc1._fuse_lora(lora_scale)
                    mlp_module.fc2._fuse_lora(lora_scale)

        if fuse_text_encoder:
            if hasattr(self, "text_encoder"):
                fuse_text_encoder_lora(self.text_encoder)
            if hasattr(self, "text_encoder_2"):
                fuse_text_encoder_lora(self.text_encoder_2)

        if fuse_unet:
            for attn_processor_name, attn_processor in self.unet.attn_processors.items():
                attn_module = self.unet
                for n in attn_processor_name.split(".")[:-1]:
                    attn_module = getattr(attn_module, n)

                if "attn1.processor" in attn_processor_name:
                    attn_module.to_q.set_lora_layer(attn_module.processor.to_q_lora)
                    attn_module.to_k.set_lora_layer(attn_module.processor.to_k_lora)
                    attn_module.to_v.set_lora_layer(attn_module.processor.to_v_lora)
                    attn_module.to_out[0].set_lora_layer(attn_module.processor.to_out_lora)
                elif "attn2.processor" in attn_processor_name:
                    attn_module.to_k_image = copy.deepcopy(attn_module.to_k)
                    attn_module.to_v_image = copy.deepcopy(attn_module.to_v)

                    attn_module.to_q.set_lora_layer(attn_module.processor.to_q_lora)
                    attn_module.to_k.set_lora_layer(attn_module.processor.to_k_lora)
                    attn_module.to_v.set_lora_layer(attn_module.processor.to_v_lora)
                    attn_module.to_out[0].set_lora_layer(attn_module.processor.to_out_lora)

                    attn_module.to_k_image.set_lora_layer(attn_module.processor.to_k_lora_image)
                    attn_module.to_v_image.set_lora_layer(attn_module.processor.to_v_lora_image)

            self.unet.fuse_lora(lora_scale)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        images_ref: Optional[Dict[str, Any]] = None,    # custom param
        start_merge_step: int = 10,                     # custom param
        fuse_scale: float = 0.8,                        # custom param
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        # for delay conditioning
        prompt = prompt.replace("<|subj|>", "")
        (
            prompt_embeds_pure,
            negative_prompt_embeds_pure,
            pooled_prompt_embeds_pure,
            negative_pooled_prompt_embeds_pure,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # for normal conditioning
        object_pixel_values = images_ref["object_pixel_values"].unsqueeze(0).to(device=device, dtype=self.image_encoder.dtype)
        image_token_mask = images_ref["image_token_mask"].to(device)
        image_token_idx_mask = images_ref["image_token_idx_mask"].to(device)

        object_embeds = self.image_encoder(object_pixel_values)
        object_embeds, object_embeds_2 = self.obj_embeds_proj(object_embeds)

        self.object_infos["object_embeds"] = object_embeds[image_token_idx_mask]
        self.object_infos["image_token_mask"] = image_token_mask
        self.object_infos_2["object_embeds"] = object_embeds_2[image_token_idx_mask]
        self.object_infos_2["image_token_mask"] = image_token_mask

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds_pure = pooled_prompt_embeds_pure
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        face_embeds = images_ref["face_embeds"].to(device=device, dtype=self.unet.dtype)
        uncond_face_embeds = torch.zeros_like(face_embeds)

        face_pixel_values = images_ref["face_pixel_values"].to(device=device, dtype=self.image_encoder.dtype)
        clip_face_embeds = self.image_encoder_bak(face_pixel_values, bak=True)
        uncond_clip_face_embeds = torch.zeros_like(clip_face_embeds)

        face_embeds = self.face_embeds_proj(face_embeds, clip_face_embeds)
        face_embeds = face_embeds.reshape(batch_size, -1, face_embeds.shape[-1])

        uncond_face_embeds = self.face_embeds_proj(uncond_face_embeds, uncond_clip_face_embeds)
        uncond_face_embeds = uncond_face_embeds.reshape(batch_size, -1, uncond_face_embeds.shape[-1])

        uncond_image_token_idx_mask = torch.zeros_like(image_token_idx_mask, dtype=bool)

        if do_classifier_free_guidance:
            prompt_embeds_pure = torch.cat([negative_prompt_embeds_pure, prompt_embeds_pure], dim=0)
            add_text_embeds_pure = torch.cat([negative_pooled_prompt_embeds_pure, add_text_embeds_pure], dim=0)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)

            face_embeds = torch.cat([uncond_face_embeds, face_embeds], dim=0)
            image_token_idx_mask = torch.cat([uncond_image_token_idx_mask, image_token_idx_mask], dim=0)

            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds_pure = prompt_embeds_pure.to(device)
        add_text_embeds_pure = add_text_embeds_pure.to(device)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)

        face_embeds = face_embeds.to(device)

        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                if i < start_merge_step:
                    added_cond_kwargs = {"text_embeds": add_text_embeds_pure, "time_ids": add_time_ids}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states={
                            "encoder_hidden_states": prompt_embeds_pure,
                            "image_hidden_states": face_embeds,
                            "image_token_idx_mask": image_token_idx_mask,
                            "fuse_scale": 0.0,
                        },
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states={
                            "encoder_hidden_states": prompt_embeds,
                            "image_hidden_states": face_embeds,
                            "image_token_idx_mask": image_token_idx_mask,
                            "fuse_scale": fuse_scale,
                        },
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
