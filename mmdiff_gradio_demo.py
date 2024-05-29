import os
import random
from glob import glob

import numpy as np
import cv2
import torch
from torchvision.io import read_image, ImageReadMode

from diffusers import AutoencoderKL
from transformers import AutoTokenizer, CLIPImageProcessor
from mmdiff import MMDiffStableDiffusionXLPipeline
from datasets import PersonalizedDataset

import spaces
import gradio as gr


# global variable
MAX_SEED = np.iinfo(np.int32).max

vae_path = "madebyollin/sdxl-vae-fp16-fix"
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "openai/clip-vit-large-patch14"
mmdiff_ckpt = "checkpoints/portrait_generation"

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = CLIPImageProcessor()

tokenizer_one = AutoTokenizer.from_pretrained(
    base_model_path,
    subfolder="tokenizer",
    use_fast=False,
    local_files_only=True,
)

tokenizer_two = AutoTokenizer.from_pretrained(
    base_model_path,
    subfolder="tokenizer_2",
    use_fast=False,
    local_files_only=True,
)

vae = AutoencoderKL.from_pretrained(
    vae_path,
    subfolder=None,
    torch_dtype=torch.float16,
    local_files_only=True
)

pipeline = MMDiffStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    vae=vae,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    variant="fp16",
    torch_dtype=torch.float16,
    local_files_only=True,
).to(device)

pipeline.load_from_checkpoint(image_encoder_path, mmdiff_ckpt, device)


@spaces.GPU(enable_queue=True)
def generate_image(upload_images, prompt, negative_prompt, num_inference_steps, num_inference_images, fuse_scale, seed, progress=gr.Progress(track_tqdm=True)):
    # check the trigger word
    if "<|subj|>" not in prompt:
        raise gr.Error(f"Cannot find the trigger word '<|subj|>' in text prompt! Please refer to step 2️⃣")

    if upload_images is None:
        raise gr.Error(f"Cannot find any input image! Please refer to step 1️⃣")

    tv_input_images, cv_input_images = [], []
    for upload_image in upload_images:
        tv_input_image, cv_input_image = load_image(upload_image)
        tv_input_images.append(tv_input_image)
        cv_input_images.append(cv_input_image)
    
    inference_dataset = PersonalizedDataset(
        prompt=prompt,
        images_ref_root=None,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        image_processor=image_processor,
        max_num_objects=1,
    )
    images_ref = inference_dataset.prepare_data(tv_input_images, cv_input_images)

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
    images = []
    for _ in range(num_inference_images):
        image = pipeline(
            prompt=prompt,
            images_ref=images_ref,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            start_merge_step=0,
            fuse_scale=fuse_scale,
            guidance_scale=5.0,
            generator=generator,
            height=512,
            width=512,
        ).images[0]
        images.append(image)
    return images, gr.update(visible=True)


def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)


def upload_example_to_gallery(images, prompt, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)


def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    

def remove_tips():
    return gr.update(visible=False)


def load_image(path):
    tv_image = read_image(path, mode=ImageReadMode.RGB)
    cv_image = cv2.imread(path)
    return tv_image, cv_image


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def get_image_path_list(folder_name):
    image_paths = sorted(
        glob(os.path.join(folder_name, "*.jpg")) + \
        glob(os.path.join(folder_name, "*.png")) + \
        glob(os.path.join(folder_name, "*.jpeg"))
    )
    return image_paths


def get_example():
    case = [
        [
            get_image_path_list('./demo_data/Barack Obama'),
            "a man<|subj|> in front of the White House",
            "longbody, lowres, bad anatomy, bad teeth, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ],
        [
            get_image_path_list('./demo_data/Wang Ou'),
            "a woman<|subj|> in front of the White House",
            "longbody, lowres, bad anatomy, bad teeth, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ],
    ]
    return case


### Description and style
title = r"""
<h1 align="center">MM-Diff: High-Fidelity Image Personalization via Multi-Modal Condition Integration</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://mm-diff.github.io/' target='_blank'><b>MM-Diff: High-Fidelity Image Personalization via Multi-Modal Condition Integration</b></a>.<br>
❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
1️⃣ Upload images of someone you want to customize. Multiple images are also supported.<br>
2️⃣ Enter a text prompt, making sure to <b>follow the class word</b> you want to customize with the <b>trigger word</b>: `<|subj|>`, such as: `man<|subj|>` or `woman<|subj|>` or `girl<|subj|>`.<br>
3️⃣ Click the <b>Submit</b> button to start customizing.
"""

article = r"""

If MM-Diff is helpful, please help to ⭐ the <a href='https://github.com/alibaba/mm-diff' target='_blank'>Github Repo</a>. Thanks! 
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:

```bibtex
@article{wei2024mm,
  title={MM-Diff: High-Fidelity Image Personalization via Multi-Modal Condition Integration},
  author={Wei, Zhichao and Su, Qingkun and Qin, Long and Wang, Weizhi},
  journal={arXiv preprint arXiv:2403.15059},
  year={2024}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>weizhichao.wzc@alibaba-inc.com</b>.
"""

tips = r"""
### Usage tips of MM-Diff
1. Upload more photos of the person to be customized to improve ID fidelty. If the input is Asian face(s), maybe consider adding 'asian' before the class word, e.g., `asian man<|subj|>`.
2. Adjust 'fuse_scale' to get a trade-off between subject fidelity and text fidelity. 0.8 works for most cases.
"""

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="Drag (Select) 1 or more photos of your face",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                       info="Try something like 'a photo of a man/woman<|subj|>', '<|subj|>' is the trigger word.",
                       placeholder="A photo of a [man/woman<|subj|>]...")
            submit = gr.Button("Submit")

            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                )
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=50,
                    step=1,
                    value=25,
                )
                num_outputs = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=8,
                )
                fuse_scale = gr.Slider( 
                    label="Weight of image condition",
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.8,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=23,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
            usage_tips = gr.Markdown(label="Usage tips of MM-Diff", value=tips ,visible=False)

        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[files, prompt, negative_prompt, num_steps, num_outputs, fuse_scale, seed],
            outputs=[gallery, usage_tips]
        )

    gr.Examples(
        examples=get_example(),
        inputs=[files, prompt, negative_prompt],
        run_on_click=True,
        fn=upload_example_to_gallery,
        outputs=[uploaded_files, clear_button, files],
    )
    
    gr.Markdown(article)
    
demo.launch(share=False)
