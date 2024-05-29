import os
import json
import math
import random
import argparse
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision

import spacy
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    CLIPTextModelWithProjection,
)
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import load_model
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import build_sam, SamPredictor


class CLIP:
    def __init__(self, model_name_or_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )

        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        outputs = self.model(
            input_ids=inputs["input_ids"].to(self.model.device),
            attention_mask=inputs["attention_mask"].to(self.model.device)
        )

        return outputs.text_embeds


class BLIP:
    def __init__(self, model_name_or_path, device):
        # Salesforce/blip-image-captioning-large
        self.processor = BlipProcessor.from_pretrained(model_name_or_path)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device, torch.float16)

        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text


class BLIP2:
    def __init__(self, model_name_or_path, device):
        # Salesforce/blip-image-captioning-large
        self.processor = Blip2Processor.from_pretrained(model_name_or_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, image, prompt):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device, torch.float16)

        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text


class PersonalizedGroundingDINO:
    def __init__(self, cfg_path, model_name_or_path, device):
        self.model = load_model(
            model_config_path=cfg_path,
            model_checkpoint_path=model_name_or_path,
            device=device,
        )
        self.model.to(device)

        self.image_transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.device = device
        self.box_threshold = 0.3
        self.text_threshold = 0.25
    
    @torch.no_grad()
    def __call__(self, image, caption):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        image = self.image_transforms(image, None)[0].to(self.device)
        outputs = self.model(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # filter outputs
        mask = logits.max(dim=1)[0] > self.box_threshold
        logits = logits[mask]
        boxes = boxes[mask]

        # get phrase
        tokenized = self.model.tokenizer(caption)

        pred_phrases = []
        for logit in logits:
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, self.model.tokenizer)
            pred_phrases.append(pred_phrase)
        
        return pred_phrases, boxes, logits.max(dim=1)[0]


class DataProcessor:
    def __init__(self, device):
        # BLIP init
        self.blip2 = BLIP2(
            model_name_or_path="Salesforce/blip2-opt-2.7b",
            device=device,
        )

        # GroundingDINO init
        self.grounding_dino = PersonalizedGroundingDINO(
            cfg_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
            model_name_or_path="checkpoints/groundingdino_swinb_cogcoor.pth",
            device=device,
        )

        # SAM init
        sam = build_sam(checkpoint="checkpoints/sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(sam.to(device))

        # spacy init
        self.nlp = spacy.load("en_core_web_sm")

        # hyper-parameters
        self.iou_threshold = 0.5
        self.color = [[255, 0, 0], [0, 0, 255]]

        self.device = device

    def visualize_single_image(self, image, boxes, masks, pred_phrases, save_path):
        image_vis = np.ascontiguousarray(np.array(image)[:, :, ::-1])

        for i in range(len(pred_phrases)):
            cv2.rectangle(image_vis, boxes[i, :2], boxes[i, 2:], color=self.color[i], thickness=2)
            cv2.putText(image_vis, pred_phrases[i], (boxes[i, 0], boxes[i, 1] - 10), color=self.color[i], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2)
            image_vis[masks[i]] = image_vis[masks[i]] * 0.4 + np.array([self.color[i]]) * 0.6

        cv2.imwrite(save_path, image_vis)

    def __call__(self, data_path):
        with open(os.path.join(data_path, "train_list.txt"), "r") as f:
            image_infos = f.read().splitlines()
        
        image_infos = random.sample(image_infos, 400000)    # 400k samples

        image_ids = []
        image_id = 0
        for image_info in tqdm(image_infos, desc=f"Processing images"):
            image_path = image_info.split(" ")[0]
            
            # Label the image step by step
            image_path = os.path.join(data_path, image_path)
            image = Image.open(image_path).convert("RGB")
            image_save = deepcopy(image)

            # 1. generate caption
            caption = self.blip2(image, prompt=None)[0].strip()

            caption_noun_phrases = []
            caption_noun_phrases_idx = []
            
            for chunk in self.nlp(caption).noun_chunks:
                if not chunk.text.split(" ")[-1].endswith("ing"):
                    caption_noun_phrases.append(chunk.text)
                    caption_noun_phrases_idx.append((chunk.start_char, chunk.end_char))                

            # 2. generate object bounding boxes
            pred_phrases, boxes, logits = self.grounding_dino(image, ",".join(caption_noun_phrases))

            img_w, img_h = image.size   # use '.size' to obtain image shape for PIL.Image object
            boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
            boxes = boxes * torch.Tensor([img_w, img_h, img_w, img_h]).to(self.device)
            
            # use NMS to filter overlapped boxes
            nms_idx = torchvision.ops.nms(boxes, logits, self.iou_threshold).cpu().numpy().tolist()
            boxes = boxes[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]

            valid_idx = []
            exclude_kws = ["two", "three", "four", "many"]
            for i, pred_phrase in enumerate(pred_phrases):
                split_pred_phrase = pred_phrase.split(" ")
                if split_pred_phrase[0] in exclude_kws:
                    continue
                else:
                    valid_idx.append(i)
            
            boxes = boxes[valid_idx]
            pred_phrases = [pred_phrases[idx] for idx in valid_idx]

            if len(pred_phrases) == 0 or len(pred_phrases) > 2:
                continue

            if len(set(pred_phrases)) != len(pred_phrases):
                continue
            
            caption_noun_phrases_valid = []
            caption_noun_phrases_idx_valid = []
            boxes_valid_idx = []
            for i, pred_phrase in enumerate(pred_phrases):
                if pred_phrase in caption_noun_phrases:
                    caption_noun_phrases_valid.append(pred_phrase)
                    caption_noun_phrases_idx_valid.append(caption_noun_phrases_idx[caption_noun_phrases.index(pred_phrase)])
                    boxes_valid_idx.append(i)
            
            if len(boxes_valid_idx) == 0:
                continue

            boxes = boxes[boxes_valid_idx]
            caption_noun_phrases = caption_noun_phrases_valid
            caption_noun_phrases_idx = caption_noun_phrases_idx_valid

            # 3. generate object mask
            image = np.array(image)
            self.sam_predictor.set_image(image)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(self.device)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            if masks.shape[0] != boxes.shape[0]:
                continue
            
            ############################ for bebug ############################
            if image_id % 100 == 0:
                self.visualize_single_image(
                    image=image,
                    boxes=boxes.cpu().numpy().astype(int),
                    masks=masks.squeeze(1).cpu().bool().numpy(),
                    pred_phrases=caption_noun_phrases,
                    save_path=os.path.join(data_path, "anno_vis", f"{image_id:09d}.jpg"),
                )
            ############################ for bebug ############################

            # 4. save as predifined format
            segmaps = torch.zeros(masks.shape[-2:])
            background_idx = 0
            for object_idx, mask in enumerate(masks):
                segmaps[mask.cpu()[0]] = background_idx + object_idx + 1
            
            save_dir = os.path.join(data_path, f"{image_id:09d}"[:5])
            os.makedirs(save_dir, exist_ok=True)

            image_save.save(os.path.join(save_dir, f"{image_id:09d}.jpg"))
            # shutil.copy(image_path, os.path.join(save_dir, f"{image_id:09d}.{image_path.split('.')[-1]}"))

            segmaps = segmaps.numpy()
            np.save(os.path.join(save_dir, f"{image_id:09d}.npy"), segmaps)

            output_json = {}
            output_json["image_id"] = f"{image_id:09d}"
            output_json["caption"] = caption
            output_json["segments"] = []
            for noun_phrase, noun_phrase_idx, box in zip(caption_noun_phrases, caption_noun_phrases_idx, boxes):
                background_idx += 1

                box = box.cpu().numpy().tolist()
                box[0] = math.ceil(box[0])
                box[1] = math.ceil(box[1])
                box[2] = math.floor(box[2])
                box[3] = math.floor(box[3])

                output_json["segments"].append({
                    "id": background_idx,
                    "word": noun_phrase,
                    "start": noun_phrase_idx[0],
                    "end": noun_phrase_idx[1],
                    "box": box,
                })
            
            with open(os.path.join(save_dir, f"{image_id:09d}.json"), "w", encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False)

            image_ids.append(f"{image_id:09d}")
            image_id += 1
        
        with open(os.path.join(data_path, f"image_ids_train.txt"), "w") as f:
            for i, image_id in enumerate(image_ids):
                if i == len(image_ids) - 1:
                    f.write(image_id)
                else:
                    f.write(image_id + "\n")

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Annotation")
    parser.add_argument("--data_path", type=str, default=None)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    data_processor = DataProcessor(device=device)
    data_processor(args.data_path)
