import gradio as gr
import numpy as np
from PIL import Image
from src.segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import requests
from src.lora import LoRA_sam
from src.segment_anything import build_sam_vit_b
import yaml
import cv2
import torch

with open("./config.yaml", "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

device = device = "cuda" if torch.cuda.is_available() else "cpu"
colors = (255, 0, 0)
# SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{sam_lora.rank}.safetensors")
sam_lora.sam.to(device)
predictor = SamPredictor(sam_lora.sam)
# Acceleration and reduction memory techniques

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            f'''# Sam LoRA loaded with the rank {sam_lora.rank} for segmenting rings. The image encoder is vit_b.
            '''
        )

    with gr.Row():
        original_img = gr.State(value=None)
        input_img = gr.Image(label="Input Image")
        mask_img = gr.Image(label="Mask")

    with gr.Row():
        with gr.Tab(label="Segmentation"):
            selected_points = gr.State([])
            masks = gr.State([])

            with gr.Row().style(equal_height=True):
                undo_points_button = gr.Button("Undo point")
                reset_points_button = gr.Button("Reset points")
                segment_button = gr.Button("Generate mask")



    def store_original_image(image):
        return image, [] # reset the selected_points
    

    def point_selection(image, selected_points, evt: gr.SelectData):
        selected_points.append(evt.index)
        print(selected_points)
        #draw points
        for point in selected_points:
            cv2.drawMarker(image, point, colors, markerSize=20, thickness=15)
        if image[..., 0][0, 0] == image[..., 2][0, 0]:  # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image) 
    
    def undo_points(original_image, selected_points):
        temp = original_image.copy()

        if len(selected_points) != 0:
            selected_points.pop()
            for point in selected_points:
                cv2.drawMarker(temp, point, colors, markerType=1, markerSize=20, thickness=15)
                
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

        return Image.fromarray(temp) 
    

    def reset_points(original_image):

        return original_image, []
    

    def generate_mask(image, selected_points):
        input_points = []
        for x,y in (selected_points):
            input_points.append(x)
            input_points.append(y)
        predictor.set_image(image)
        input_points = np.array(input_points)
        masks, _, _ = predictor.predict(
            box=input_points,
            multimask_output=False
        )
        mask =  Image.fromarray(masks[0])
        
        return mask

    input_img.upload(
        store_original_image,
        inputs=[input_img],
        outputs=[original_img, selected_points]
    )
    
    input_img.select(
        point_selection,
        inputs=[input_img, selected_points],
        outputs=[input_img]
    )

    undo_points_button.click(
        undo_points,
        inputs=[original_img, selected_points],
        outputs=[input_img]
    )
    reset_points_button.click(
        reset_points,
        inputs=[original_img],
        outputs=[input_img, selected_points]
    )

    segment_button.click(
        generate_mask, 
        inputs=[input_img, selected_points], 
        outputs=[mask_img]
    )

    if __name__ == "__main__":
        demo.launch(share=True)