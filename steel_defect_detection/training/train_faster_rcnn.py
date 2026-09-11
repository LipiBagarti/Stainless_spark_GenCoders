"""
Training script for Faster R-CNN with ResNet-50-FPN backbone on NEU-DET.

Uses PyTorch and torchvision with ground-truth bounding box annotations
from the YOLO format / XML annotations converted to standard dataset format.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class NeuDetDataset(Dataset):
    """PyTorch Dataset loading YOLO-formatted images and txt annotations for Faster R-CNN."""

    def __init__(self, img_dir: Path, label_dir: Path, transforms=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.img_files = sorted([f for f in self.img_dir.glob("*.jpg")] + [f for f in self.img_dir.glob("*.bmp")])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx = float(parts[1]) * w
                        cy = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h

                        x1 = max(0.0, cx - bw / 2)
                        y1 = max(0.0, cy - bh / 2)
                        x2 = min(float(w), cx + bw / 2)
                        y2 = min(float(h), cy + bh / 2)

                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            # In torchvision Faster R-CNN, 0 is background, 1..6 are defect classes
                            labels.append(cls_id + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # Convert image to tensor [C, H, W] in [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int = 7):
    # num_classes = 6 defect classes + 1 background class
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_faster_rcnn(config_path: str = "configs/config.yaml", epochs: int = 20, batch_size: int = 4):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = ROOT / "data" / "yolo_format"
    train_dataset = NeuDetDataset(data_dir / "images" / "train", data_dir / "labels" / "train")
    val_dataset = NeuDetDataset(data_dir / "images" / "val", data_dir / "labels" / "val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(num_classes=7)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.33)

    output_dir = ROOT / "models" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(f"Starting Faster R-CNN training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = output_dir / "faster_rcnn_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best Faster R-CNN checkpoint to {save_path}")

    print("Faster R-CNN training complete.")


if __name__ == "__main__":
    train_faster_rcnn()
