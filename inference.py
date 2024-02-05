import argparse
import torch
import numpy as np
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image
from pathlib import Path
from os.path import split

parser = argparse.ArgumentParser(description="SAM-fine-tune Inference")
parser.add_argument("image", help="The file to perform inference on.")
parser.add_argument("-o", "--output", required=True, help="File to save the inference to.")
parser.add_argument("-r", "--rank", default=512, help="LoRA model rank.")
parser.add_argument("-l", "--lora", default="lora_weights/lora_rank512.safetensors", help="Location of LoRA Weight file.")
parser.add_argument("-d", "--device", choices=["cuda", "cpu"], default="cuda", help="What device to run the inference on.")
parser.add_argument("-b", "--baseline", action="store_true", help="Use baseline SAM instead of a LoRA model.")
parser.add_argument("-m", "--mask", default=None, help="Location of the mask file to use for inference.")

args = parser.parse_args()


def inference_model(image_path, save_name, mask_path):
    image = Image.open(image_path)

    if mask_path:
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask = np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
    else:
        w, h = image.size
        box = [0, 0, w, h]

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)

    if args.baseline:
        model = sam
    else:
        rank = args.rank
        sam_lora = LoRA_sam(sam, rank)
        sam_lora.load_lora_parameters(args.lora)
        model = sam_lora.sam

    model.eval()
    model.to(device)
    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(box),
        multimask_output=False
    )
    plt.imsave(save_name, masks[0])
    print("IoU Prediction:", iou_pred[0])


input_file = args.image
output_file = args.output

output_path, _ = split(output_file)
if output_path:
    Path(output_path).mkdir(parents=True, exist_ok=True)

if args.device == "cuda":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

inference_model(input_file, output_file, mask_path=args.mask)
