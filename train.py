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
from src.metrics import calculate_iou
"""
This file is used to train a LoRA_sam model. I use that monai DiceLoss for the training. The batch size and number of epochs are taken from the configuration file.
The model is saved at the end as a safetensor.
"""
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
val_dataset_path = config_file["DATASET"].get("VAL_PATH")
# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam
# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")

if val_dataset_path:
  val_ds = DatasetSegmentation(config_file, processor, mode="val")
# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
if val_dataset_path:
  val_dataloader = DataLoader(val_ds, batch_size=config_file["VAL"]["BATCH_SIZE"] if config_file.get("VAL") else config_file["TRAIN"]["BATCH_SIZE"],
                              shuffle=True,
                              collate_fn=collate_fn)
# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=config_file["TRAIN"]["LEARNING_RATE"], weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

threshold_iou = config_file["TRAIN"].get("THR_IOU", 0.5)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

total_loss = []
iou_metrics_train = []
iou_metrics_val = []

for epoch in range(num_epochs):
    epoch_losses = []

    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
      
      outputs = model(batched_input=batch,
                      multimask_output=False)

      stk_gt, stk_out = utils.stacking_batch(batch, outputs)
      stk_out = stk_out.squeeze(1)
      stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
      loss = seg_loss(stk_out, stk_gt.float().to(device))
      
      optimizer.zero_grad()
      loss.backward()
      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

      # эти строчки я добавил
      iou = calculate_iou(stk_out, stk_gt.int().to(device), thr=threshold_iou)
      iou_metrics_train.append(iou)
    
    # эти строчки я добавил
    if val_dataset_path:
      model.eval()
      for i, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
          outputs = model(batched_input=batch,
                          multimask_output=False)

          stk_gt, stk_out = utils.stacking_batch(batch, outputs)
          stk_out = stk_out.squeeze(1)
          stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]

          # эти строчки я добавил
          iou = calculate_iou(stk_out, stk_gt.int().to(device), thr=threshold_iou)
          iou_metrics_val.append(iou)

    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')
    print(f'IoU Train: {mean(iou_metrics_train)}')
    print(f'IoU Val: {mean(iou_metrics_val)}')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
