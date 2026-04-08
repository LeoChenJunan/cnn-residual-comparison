"""
Model B: CNN with Residual Block + Data Augmentation + Dropout
本模型在 Model A 的基礎上加入正則化技巧，
用來觀察是否能降低 overfitting 並提升泛化能力。
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# 固定 random seed，確保每次實驗結果可重現
torch.manual_seed(42)
np.random.seed(42)

# ---------------- DATASET ----------------
class CIFARJsonDataset(Dataset):
    def __init__(self, path, label_map=None, augment=False):
        """
        自訂 Dataset，負責：
        1. 讀取 json 格式的 CIFAR-like 資料
        2. 做影像正規化與維度轉換
        3. 在訓練階段選擇性加入 Data Augmentation
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 將影像轉為 numpy array，並轉成 float32
        self.images = np.array([d["Image"] for d in raw], dtype=np.float32)

        # 將 pixel 值從 [0, 255] 正規化到 [-1, 1]
        self.images = self.images / 255.0
        self.images = (self.images - 0.5) / 0.5

        # 將資料格式轉為 PyTorch CNN 慣用的 (C, H, W)
        self.images = self.images.transpose(0, 3, 1, 2)

        # 判斷是否包含標籤（train / val 有，test 沒有）
        self.has_label = "Label" in raw[0]
        if self.has_label:
            raw_labels = np.array([d["Label"] for d in raw], dtype=np.int64)
            self.labels = np.array([label_map[l] for l in raw_labels], dtype=np.int64)

        # 是否啟用 Data Augmentation
        # 只會在 training set 設為 True，避免資料洩漏
        self.augment = augment

    def __len__(self):
        # 回傳資料集大小
        return len(self.images)

    def __getitem__(self, idx):
        # 取出單張影像並轉為 Tensor
        x = torch.tensor(self.images[idx], dtype=torch.float32)

        # -------- Data Augmentation（Model B 專屬）--------
        # 僅在訓練階段啟用，用來增加資料多樣性
        if self.augment:
            # 隨機水平翻轉（模擬左右對稱變化）
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])

            # 隨機裁切（padding + crop）
            # 用來模擬影像平移，提高模型的泛化能力
            pad = 4
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad))
            i = torch.randint(0, pad * 2, (1,)).item()
            j = torch.randint(0, pad * 2, (1,)).item()
            x = x[:, i:i+32, j:j+32]

        if self.has_label:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        else:
            # test set 不包含標籤
            return x


# ---------------- MODEL ----------------
class ResidualBlock(nn.Module):
    """
    標準 Residual Block：
    - 兩層 Conv + BatchNorm
    - Identity shortcut 直接相加
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # shortcut
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # 殘差相加
        return torch.relu(out)


class ModelB(nn.Module):
    """
    Model B 架構：
    - 與 Model A 完全相同的 CNN + Residual Blocks
    - 唯一差異：在分類器前加入 Dropout
    """
    def __init__(self, num_classes):
        super().__init__()

        # 初始卷積層，負責提取低階特徵
        self.stem = nn.Conv2d(3, 64, 3, padding=1)

        # 兩個 Residual Blocks
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)

        # Global Average Pooling，減少參數量
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout（僅訓練時啟用）
        self.dropout = nn.Dropout(p=0.5)

        # 最終分類層
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 正則化，降低 overfitting
        return self.fc(x)


# ---------------- TRAIN / EVAL ----------------
def run_epoch(model, loader, criterion, optimizer=None):
    """
    單一 epoch 的訓練或驗證流程：
    - 若 optimizer 存在 → training mode（Dropout ON）
    - 否則 → evaluation mode（Dropout OFF）
    """
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


# ---------------- MAIN ----------------
if __name__ == "__main__":

    # ---------- LABEL MAP ----------
    # 僅使用 training data 建立 label mapping，避免資料洩漏
    with open("train.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_labels = [d["Label"] for d in raw]
    unique_labels = sorted(set(raw_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    inv_map = {i: l for l, i in label_map.items()}
    num_classes = len(unique_labels)

    # ---------- DATASET ----------
    full_dataset = CIFARJsonDataset("train.json", label_map=label_map, augment=False)
    test_dataset = CIFARJsonDataset("test.json")

    # Train / Validation split（80 / 20）
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Model B：僅訓練集啟用 Data Augmentation
    train_set.dataset.augment = True

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # ---------- MODEL ----------
    model = ModelB(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    # ---------- TRAIN ----------
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        print(
            f"Epoch {ep:03d} | "
            f"Train Loss={tl:.4f}, Acc={ta:.4f} | "
            f"Val Loss={vl:.4f}, Acc={va:.4f}"
        )

    # ---------- OUTPUT ----------
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy curve
    plt.plot(history["train_acc"], label="Train accuracy")
    plt.plot(history["val_acc"], label="Validation accuracy")
    plt.title("Model B_Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "output_accuracy.png"))
    plt.close()

    # Loss curve
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("Model B_Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "output_loss.png"))
    plt.close()

    # 記錄實驗設定與最終結果
    with open(os.path.join(out_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump({
            "Learning rate": 1e-3,
            "Epoch": epochs,
            "Batch size": 64,
            "Final train accuracy": history["train_acc"][-1],
            "Validation accuracy": history["val_acc"][-1],
            "Final train loss": history["train_loss"][-1],
            "Final validation loss": history["val_loss"][-1]
        }, f, indent=4)

    # ---------- TEST PREDICTION ----------
    model.eval()
    preds = []
    with torch.no_grad():
        for x in DataLoader(test_dataset, batch_size=64):
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())

    preds = [inv_map[p] for p in preds]

    with open(os.path.join(out_dir, "test_set_prediction.json"), "w", encoding="utf-8") as f:
        json.dump({"Predictions": preds}, f, indent=4)

    print("Model B done. All outputs generated.")
