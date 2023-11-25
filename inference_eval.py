import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import monai
import numpy as np
"""
This file compute the evaluation metric (Dice cross entropy loss) for all trained LoRA SAM with different ranks. This gives the plot that is in ./plots/rank_comparison.jpg
which compares the performances on test the test set.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)
rank_list = [2, 4, 6, 8, 16, 32, 64, 128, 256, 512]
rank_loss = []
total_baseline_loss = []
baseline_loss = 0
# Load SAM model
with torch.no_grad():
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    baseline = sam
    processor = Samprocessor(baseline)
    dataset = DatasetSegmentation(config_file, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    baseline.eval()
    baseline.to(device)   
    for i, batch in enumerate(tqdm(test_dataloader)):
        
        outputs = baseline(batched_input=batch,
            multimask_output=False)
        
        gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
        loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))

        total_baseline_loss.append(loss.item())
        

    print(f'Mean dice score: {mean(total_baseline_loss)}')
    baseline_loss = mean(total_baseline_loss)

    for rank in rank_list:
        sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
        baseline = sam
        #Create SAM LoRA
        sam_lora = LoRA_sam(sam, rank)
        sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{rank}.safetensors")  
        model = sam_lora.sam
        
        # Process the dataset
        processor = Samprocessor(model)
        dataset = DatasetSegmentation(config_file, processor, mode="test")

        # Create a dataloader
        test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)


        # Set model to train and into the device
        model.eval()
        model.to(device)
    

        total_score = []
        for i, batch in enumerate(tqdm(test_dataloader)):
            
            outputs = model(batched_input=batch,
                multimask_output=False)
            
            gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
            loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))

            total_score.append(loss.item())
            

        print(f'Mean dice score: {mean(total_score)}')
        rank_loss.append(mean(total_score))


print("RANK LOSS :", rank_loss)

width = 0.25  # the width of the bars
multiplier = 0
models_results= {"Baseline": baseline_loss,
                 "Rank 2": rank_loss[0], 
                 "Rank 4": rank_loss[1], 
                 "Rank 6": rank_loss[2],
                 "Rank 8": rank_loss[3],
                 "Rank 16": rank_loss[4],
                 "Rank 32": rank_loss[5],
                 "Rank 64": rank_loss[6],
                 "Rank 128": rank_loss[7],
                 "Rank 256": rank_loss[8],
                 "Rank 512": rank_loss[9]
                 }
eval_scores_name = ["Rank"]
x = np.arange(len(eval_scores_name))
fig, ax = plt.subplots(layout='constrained')

for model_name, score in models_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=model_name)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dice Loss')
ax.set_title('LoRA trained on 50 epochs - Rank comparison on test set')
ax.set_xticks(x + width, eval_scores_name)
ax.legend(loc=3, ncols=2)
ax.set_ylim(0, 0.2)

plt.savefig("./plots/rank_comparison.jpg")