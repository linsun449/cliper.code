import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import gradio as gr
import numpy as np
import torch
from PIL import Image
from ovs.pipeline import Pipeline
from util.tools import load_yaml
import torch.nn.functional as F

image_files = {
    "assets/natural_house.jpg": ["house;sky;grass;wall", "", 0.4, 0.5],
    "assets/natural_dog.jpg": ["dog;sod;chair", "", 0.55, 0.5],
    "assets/painting_bicycle.jpg": ["bicycle;door;flower", "", 0.75, 0.5],
    "assets/clipart_dragon.jpg": ["dragon;princess;armour man", "", 0.5, 0.5],
    "assets/sketch_fence.jpg": ["fence;grass;tree;mountain", "", 0.4, 0.5],
}

PALETTE = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_yaml("scripts/config/app.yaml")
cfg.semantic_templates = [line.strip() for line in list(open(cfg.semantic_templates))]
CLIPer = Pipeline(cfg)
embedding = torch.load("embeddings_large14.pth", map_location=device)
print(cfg, CLIPer.attn_refine)


def test_selected_image(file_path):
    file_path = f"assets/{file_path}.jpg"
    target, background_class, segmentation_threshold, class_threshold = image_files[file_path]
    return (process_image(np.array(Image.open(file_path)), target, background_class,
                          segmentation_threshold, class_threshold),
            target, background_class, segmentation_threshold, class_threshold)


def process_image(image: np.ndarray, target: str, background_class: str, segmentation_threshold: float,
                  class_threshold: float):
    bg_classes = background_class.split(";")
    fg_classes = target.split(";")
    ori_img = torch.tensor(image.astype(float)).permute(2, 0, 1) / 255.
    image = CLIPer.cliper.preprocess(Image.fromarray(image))[None]

    fg_text_features = CLIPer.cliper.classifier(fg_classes, cfg.semantic_templates)
    bg_text_features = CLIPer.cliper.classifier(bg_classes, cfg.semantic_templates)
    bg_text_features = torch.cat([bg_text_features,
                                  embedding[(fg_text_features @ embedding.T).sort().indices[:, ::1500]].flatten(0, 1)],
                                 dim=0)
    pred_mask, final_score = CLIPer(ori_img.to(device), image.to(device), fg_text_features, bg_text_features)
    pred_mask = F.interpolate(pred_mask[None], size=(ori_img.shape[-2], ori_img.shape[-1]), mode='bilinear')[0]
    pred_mask[final_score < class_threshold] = 0
    pred_mask = torch.cat([torch.ones_like(pred_mask) * segmentation_threshold, pred_mask])
    mask = pred_mask.argmax(dim=0)
    return PALETTE[mask.cpu().numpy()]


with gr.Blocks() as demo:
    test_btn = []
    with gr.Row():
        image_upload = gr.Image(label="上传图像", height=400, width=800)
        with gr.Row():
            with gr.Column():
                target_input = gr.Textbox(label="输入目标", placeholder="英文类别，以分号结尾")
                background_class_input = gr.Textbox(label="输入背景", placeholder="英文类别，以分号结尾")
                segmentation_threshold_slider = gr.Slider(0, 1, label="分割阈值", value=0.5, step=0.01,
                                                          interactive=True)
                class_threshold_slider = gr.Slider(0, 1, label="类别阈值", value=0.5, step=0.01, interactive=True)
                process_button = gr.Button("处理图像")
    with gr.Row():
        for file in image_files.keys():
            with gr.Row():
                with gr.Column():
                    test_image = gr.Image(file, label=file.split('/')[1][:-4], height=250, width=400)
                    test_button = gr.Button(file.split('/')[1][:-4], min_width=100)
                    test_btn.append(test_button)
    result_image = gr.Image(label="分割图像", interactive=False, )

    process_button.click(fn=process_image, inputs=[image_upload, target_input, background_class_input,
                                                   segmentation_threshold_slider,
                                                   class_threshold_slider], outputs=result_image)
    for label, test_button in zip(image_files.keys(), test_btn):
        test_button.click(fn=test_selected_image,
                          inputs=[test_button],
                          outputs=[result_image, target_input, background_class_input,
                                   segmentation_threshold_slider, class_threshold_slider])
demo.launch(share=True)
