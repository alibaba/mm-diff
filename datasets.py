import os
from glob import glob
from collections import OrderedDict

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode

from insightface.app import FaceAnalysis
from insightface.utils import face_align


class PadToSquare(torch.nn.Module):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h == w:
            return image
        elif h > w:
            padding = (h - w) // 2
            image = nn.functional.pad(image, (padding, padding, 0, 0), self.padding_mode, self.fill)
        else:
            padding = (w - h) // 2
            image = nn.functional.pad(image, (0, 0, padding, padding), self.padding_mode, self.fill)
        return image


def prepare_image_token_idx(image_token_mask, max_num_objects):
    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat([image_token_idx, torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long)])
        image_token_idx_mask = torch.cat([image_token_idx_mask, torch.zeros(max_num_objects - len(image_token_idx_mask), dtype=torch.bool)])
    
    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    return image_token_idx, image_token_idx_mask


class PersonalizedDataset():
    def __init__(
        self,
        prompt,
        images_ref_root,
        tokenizer_one,
        tokenizer_two,
        image_processor,
        trigger_token="<|subj|>",
        max_num_objects=2,
    ):
        self.prompt = prompt
        self.images_ref_root = images_ref_root
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.image_processor = image_processor
        self.object_transforms = nn.Sequential(OrderedDict([
            ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
            ("resize", T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True)),
            ("convert_to_float", T.ConvertImageDtype(torch.float32)),
        ]))
        self.trigger_token = trigger_token
        self.max_num_objects = max_num_objects

        tokenizer_one.add_tokens([trigger_token], special_tokens=True)
        tokenizer_two.add_tokens([trigger_token], special_tokens=True)
        self.trigger_token_id = tokenizer_one.convert_tokens_to_ids(trigger_token)

        self.app_face = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_face.prepare(ctx_id=0, det_size=(640, 640))

    def set_prompt(self, prompt):
        self.prompt = prompt
    
    def set_images_ref_root(self, images_ref_root):
        self.images_ref_root = images_ref_root

    def _tokenize_and_mask_noun_phrases_ends(self):
        input_ids = self.tokenizer_one.encode(self.prompt)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_index = 0

        for input_id in input_ids:
            if input_id == self.trigger_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_index += 1
        
        max_len = self.tokenizer_one.model_max_length
        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (max_len - len(noun_phrase_end_mask))
        
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return noun_phrase_end_mask.unsqueeze(0)
    
    def prepare_data(self, tv_input_images=None, cv_input_images=None):
        object_pixel_values, face_embeds, face_pixel_values = [], [], []

        if tv_input_images is not None and cv_input_images is not None:
            for tv_input_image, cv_input_image in zip(tv_input_images, cv_input_images):
                object_pixel_values.append(self.object_transforms(tv_input_image))

                faces = self.app_face.get(cv_input_image)
                face_embeds.append(torch.from_numpy(faces[0].normed_embedding))

                face_image = face_align.norm_crop(cv_input_image, landmark=faces[0].kps, image_size=224)
                face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                face_pixel_values.append(self.image_processor(face_image, return_tensors="pt").pixel_values[0])
        else:
            images_ref_path = sorted(
                glob(os.path.join(self.images_ref_root, "*.jpg")) + \
                glob(os.path.join(self.images_ref_root, "*.png")) + \
                glob(os.path.join(self.images_ref_root, "*.jpeg"))
            )

            for image_ref_path in images_ref_path:
                object_pixel_values.append(self.object_transforms(read_image(image_ref_path, mode=ImageReadMode.RGB)))

                image_ref = cv2.imread(image_ref_path)
                faces = self.app_face.get(image_ref)
                face_embeds.append(torch.from_numpy(faces[0].normed_embedding))

                face_image = face_align.norm_crop(image_ref, landmark=faces[0].kps, image_size=224)
                face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                face_pixel_values.append(self.image_processor(face_image, return_tensors="pt").pixel_values[0])
        
        object_pixel_values = torch.stack(object_pixel_values)
        object_pixel_values = object_pixel_values.to(memory_format=torch.contiguous_format).float()
        face_embeds = torch.stack(face_embeds)
        face_embeds = face_embeds.to(memory_format=torch.contiguous_format).float()
        face_pixel_values = torch.stack(face_pixel_values)
        
        image_token_mask = self._tokenize_and_mask_noun_phrases_ends()
        image_token_idx, image_token_idx_mask = prepare_image_token_idx(image_token_mask, max_num_objects=self.max_num_objects)

        num_objects = image_token_idx_mask.sum()

        return {
            "object_pixel_values": object_pixel_values[0:1],
            "face_embeds": face_embeds,
            "face_pixel_values": face_pixel_values,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "num_objects": num_objects,
        }
