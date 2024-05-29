import math
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from transformers.models.clip.modeling_clip import CLIPPreTrainedModel, CLIPModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def replace_text_encoder_forward(text_encoder, fuse_module, object_infos):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)

        if object_infos["image_token_mask"] is not None:
            fused_embeds = torch.cat([inputs_embeds[object_infos["image_token_mask"]], object_infos["object_embeds"]], dim=-1)
            inputs_embeds[object_infos["image_token_mask"]] = fuse_module(fused_embeds)

        embeddings = inputs_embeds + position_embeddings

        return embeddings

    text_encoder.text_model.embeddings.old_forward = text_encoder.text_model.embeddings.forward
    text_encoder.text_model.embeddings.forward = types.MethodType(forward, text_encoder.text_model.embeddings)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class CLIPEmbedsProjHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim_1, out_dim_2):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.proj_1 = nn.Linear(hidden_dim, out_dim_1)
        self.proj_2 = nn.Linear(hidden_dim, out_dim_2)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.fc(x)
        x = self.act_fn(x)

        out1 = self.proj_1(x)
        out2 = self.proj_2(x)

        return out1, out2


class FaceEmbedsProjHead(nn.Module):
    def __init__(self, face_embeds_dim=512, clip_embeds_dim=1024, corss_attention_dim=2048, num_tokens=4):
        super().__init__()
        self.corss_attention_dim = corss_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(face_embeds_dim, face_embeds_dim * 2),
            nn.GELU(),
            nn.Linear(face_embeds_dim * 2, corss_attention_dim * num_tokens)
        )

        self.layer_norm = nn.LayerNorm(corss_attention_dim)

        self.refiner = SERefiner(
            dim=corss_attention_dim,
            depth=4,
            dim_head=256,
            heads=corss_attention_dim // 256,
            embedding_dim=clip_embeds_dim,
            output_dim=corss_attention_dim,
        )
    
    def forward(self, face_embeds, clip_embeds):
        x = self.proj(face_embeds)
        x = x.reshape(-1, self.num_tokens, self.corss_attention_dim)
        x = self.layer_norm(x)

        out = self.refiner(x, clip_embeds)
        out = x + out

        return out


class CLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
        local_files_only=True,
    ):
        model = CLIPModel.from_pretrained(global_model_name_or_path, local_files_only=local_files_only)
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        return CLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values, bak=False):
        if bak:
            object_embeds = self.vision_model(object_pixel_values, output_hidden_states=True).hidden_states[-2]
        else:
            b, num_objects, c, h, w = object_pixel_values.shape

            object_pixel_values = object_pixel_values.view(b * num_objects, c, h, w)

            if h != self.image_size or w != self.image_size:
                h, w = self.image_size, self.image_size
                object_pixel_values = F.interpolate(object_pixel_values, (h, w), mode="bilinear", antialias=True)

            object_pixel_values = self.vision_processor(object_pixel_values)
            object_embeds = self.vision_model(object_pixel_values)[1]
            object_embeds = self.visual_projection(object_embeds)
            object_embeds = object_embeds.view(b, num_objects, -1)

        return object_embeds


class SEAttention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, query, dense_feats):
        dense_feats = self.norm1(dense_feats)
        query = self.norm2(query)

        b, l, _ = query.shape

        q = self.to_q(query)
        kv_input = torch.cat((dense_feats, query), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        attn = (q * scale) @ (k * scale).transpose(-2, -1)
        attn = torch.softmax(attn.float(), dim=-1).type(attn.dtype)
        out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class SERefiner(nn.Module):
    def __init__(self, dim, depth, dim_head, heads, embedding_dim, output_dim):
        super().__init__()
        
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SEAttention(dim=dim, dim_head=dim_head, heads=heads),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * 2, bias=False),
                            nn.GELU(),
                            nn.Linear(dim * 2, dim, bias=False),
                        )
                    ]
                )
            )

    def forward(self, query, dense_feats):
        dense_feats = self.proj_in(dense_feats)
        for attn, ff in self.layers:
            query = attn(query, dense_feats) + query
            query = ff(query) + query
        query = self.proj_out(query)

        return self.norm_out(query)
