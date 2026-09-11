"""
OPTIONAL second-stage classifier (EfficientNet-B0) — implements Architecture B
from docs/case_study_analysis.md Part 7.

Only use this if Part 21's experiments show YOLO's own classification head has a
measurable weakness on specific confusable classes. Not part of the default
real-time pipeline (Architecture A).

Trains on cropped defect patches extracted from YOLO-format bounding boxes, using
a folder-per-class layout:
    data/classifier_crops/train/<class_name>/*.jpg
    data/classifier_crops/val/<class_name>/*.jpg

Use preprocessing/crop_defects_for_classifier.py-style logic (build this by cropping
each YOLO label's box out of its image) to populate that folder before running this
script.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = PROJECT_ROOT / "data" / "classifier_crops"
MODELS_DIR = PROJECT_ROOT / "models"

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def build_dataloaders(batch_size: int, img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(CROPS_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(CROPS_DIR / "val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int):
    # Transfer learning from ImageNet weights — mandatory given small crop dataset
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train(epochs: int, batch_size: int, img_size: int, lr: float, patience: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_dataloaders(batch_size, img_size)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0
    MODELS_DIR.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)

        print(f"Epoch {epoch+1}/{epochs} - loss: {running_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": class_names},
                       MODELS_DIR / "efficientnet_b0_classifier.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train optional EfficientNet-B0 second-stage classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.img_size, args.lr, args.patience)
