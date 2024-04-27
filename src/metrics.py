def calculate_iou(preds, targets, thr=0.5):
    preds = (preds > thr).int()

    smooth = 1e-6
    intersection = (preds & targets).float().sum((1, 2))  # Intersection points
    union = (preds | targets).float().sum((1, 2))         # Union points

    iou = (intersection + smooth) / (union + smooth)      # IoU calculation
    return iou.mean().item()  # Returns the average IoU for the batch
