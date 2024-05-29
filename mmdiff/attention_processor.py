import torch
import torch.nn as nn

from diffusers.models.lora import LoRALinearLayer


class LoRASelfAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        lora_scale=1.0,
    ):
        super().__init__()

        self.rank = rank
        self.lora_scale = lora_scale
        
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LoRACrossAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        lora_scale=1.0,
        fuse_scale=1.0,
        num_image_tokens=4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.lora_scale = lora_scale
        self.fuse_scale = fuse_scale
        self.num_image_tokens = num_image_tokens
        
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        self.to_k_lora_image = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora_image = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if isinstance(encoder_hidden_states, dict):
            fuse_scale = encoder_hidden_states.get("fuse_scale", self.fuse_scale)
            image_token_idx_mask = encoder_hidden_states["image_token_idx_mask"]
            image_hidden_states = encoder_hidden_states["image_hidden_states"]
            encoder_hidden_states = encoder_hidden_states["encoder_hidden_states"]

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if hasattr(attn, "old_get_attention_scores"):
            attention_probs = attn.get_attention_scores(query, key, attention_mask, log_name="text")
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for image embeddings
        image_key = attn.to_k(image_hidden_states) + self.lora_scale * self.to_k_lora_image(image_hidden_states)
        image_value = attn.to_v(image_hidden_states) + self.lora_scale * self.to_v_lora_image(image_hidden_states)

        image_key = attn.head_to_batch_dim(image_key)
        image_value = attn.head_to_batch_dim(image_value)

        bsz, max_obj_num = image_token_idx_mask.shape
        if max_obj_num != image_key.shape[1] // self.num_image_tokens:
            assert max_obj_num == 1, "Input Error"
            max_obj_num = image_key.shape[1] // self.num_image_tokens
            image_token_idx_mask = image_token_idx_mask.repeat(1, max_obj_num)

        image_attn_mask = image_token_idx_mask.reshape(bsz, 1, max_obj_num, 1).repeat(1, attn.heads, 1, self.num_image_tokens)
        image_attn_mask = (~image_attn_mask.reshape(bsz*attn.heads, 1, max_obj_num*self.num_image_tokens)).float()
        invalid_mask = torch.all(image_attn_mask == 1, dim=-1).unsqueeze(-1)
        image_attn_mask = image_attn_mask.masked_fill(image_attn_mask == 1, -1e6)
        image_attn_mask = image_attn_mask.masked_fill(invalid_mask.float() == 1, 0).to(dtype=query.dtype)

        if hasattr(attn, "old_get_attention_scores"):
            image_attention_probs = attn.get_attention_scores(query, image_key, attention_mask=image_attn_mask, log_name="image")
        else:
            image_attention_probs = attn.get_attention_scores(query, image_key, attention_mask=image_attn_mask)

        image_hidden_states = torch.bmm(image_attention_probs, image_value) * (~invalid_mask).to(dtype=image_attention_probs.dtype)
        image_hidden_states = attn.batch_to_head_dim(image_hidden_states)

        hidden_states = hidden_states + fuse_scale * image_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
