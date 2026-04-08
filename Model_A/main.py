"""
Model A: CNN with Residual Block (NO Data Augmentation, NO Dropout)
本模型主要用來驗證「Residual Block 本身」對模型訓練穩定性與效能的影響
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# 固定隨機種子，確保每次實驗結果具有可重現性（方便比較模型）
torch.manual_seed(42)
np.random.seed(42)

# ---------------- DATASET ----------------
class CIFARJsonDataset(Dataset):
    def __init__(self, path, label_map=None):
        # 開啟並讀取 JSON 格式的資料集（train.json 或 test.json）
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 取出影像資料，轉為 numpy array 方便後續處理
        self.images = np.array([d["Image"] for d in raw], dtype=np.float32)

        # 將像素值從 [0,255] 正規化到 [-1,1]，有助於模型收斂
        self.images = self.images / 255.0
        self.images = (self.images - 0.5) / 0.5

        # 將影像維度由 (H, W, C) 轉成 PyTorch 慣用的 (C, H, W)
        self.images = self.images.transpose(0, 3, 1, 2)

        # 判斷資料是否包含標籤（test.json 沒有 Label）
        self.has_label = "Label" in raw[0]
        if self.has_label:
            # 取出原始標籤
            raw_labels = np.array([d["Label"] for d in raw], dtype=np.int64)
            # 使用 label_map 將標籤轉成連續整數索引
            self.labels = np.array([label_map[l] for l in raw_labels], dtype=np.int64)

    def __len__(self):
        # 回傳資料集的樣本數量
        return len(self.images)

    def __getitem__(self, idx):
        # 取出指定索引的影像資料並轉成 torch tensor
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        if self.has_label:
            # 若資料有標籤，則同時回傳影像與標籤
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        else:
            # 測試集只需要回傳影像
            return x


# ---------------- MODEL ----------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 第一層卷積，輸入與輸出 channel 數相同，確保可以做 identity shortcut
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # 第二層卷積，同樣保持輸出維度不變
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 將輸入保留下來，作為 Residual 的 shortcut
        identity = x
        # 主路徑：Conv → BN → ReLU
        out = torch.relu(self.bn1(self.conv1(x)))
        # 第二層 Conv → BN（ReLU 留到加完 shortcut 後）
        out = self.bn2(self.conv2(out))
        # 與 shortcut 做 element-wise addition
        out = out + identity
        # 最後再通過 ReLU
        return torch.relu(out)


class ModelA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 前段卷積層，將 3-channel RGB 影像轉為 64-channel 特徵圖
        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        # 中段堆疊兩個 Residual Block（深度與 Baseline 保持一致）
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        # 使用 Global Average Pooling，避免產生過多參數
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 全連接層，輸出分類結果
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 前段特徵擷取
        x = torch.relu(self.stem(x))
        # Residual Block 堆疊
        x = self.block1(x)
        x = self.block2(x)
        # 將整張特徵圖平均成 1x1
        x = self.gap(x)
        # 攤平成一維向量
        x = torch.flatten(x, 1)
        # 輸出 logits（不加 Softmax，交由 loss function 處理）
        return self.fc(x)


# ---------------- TRAIN / EVAL ----------------
def run_epoch(model, loader, criterion, optimizer=None):
    # 若 optimizer 不為 None，代表目前是訓練階段
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    # 訓練時啟用梯度，驗證時關閉以節省計算資源
    with torch.set_grad_enabled(train):
        for x, y in loader:
            # 前向傳播
            out = model(x)
            loss = criterion(out, y)

            if train:
                # 反向傳播與參數更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 累積 loss 與正確筆數
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    # 回傳平均 loss 與 accuracy
    return total_loss / total, correct / total


# ---------------- MAIN ----------------
if __name__ == "__main__":

    # ---------- LABEL MAPPING ----------
    # 讀取訓練資料以建立 label mapping
    with open("train.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_labels = [d["Label"] for d in raw]
    # 取出所有不重複的標籤
    unique_labels = sorted(set(raw_labels))
    # 將原始標籤對應成連續整數索引
    label_map = {l: i for i, l in enumerate(unique_labels)}
    inv_map = {i: l for l, i in label_map.items()}
    num_classes = len(unique_labels)

    # ---------- DATASET ----------
    # 建立完整訓練資料集與測試資料集
    full_dataset = CIFARJsonDataset("train.json", label_map=label_map)
    test_dataset = CIFARJsonDataset("test.json")

    # ---------- SPLIT ----------
    # 將訓練資料切分為訓練集與驗證集（80% / 20%）
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 建立 DataLoader 以進行 batch 訓練
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # ---------- MODEL ----------
    # 初始化模型、損失函數與最佳化器
    model = ModelA(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    # 用來記錄訓練與驗證過程中的 loss 與 accuracy
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    # ---------- TRAIN ----------
    # 進行多個 epoch 的訓練
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        # 每個 epoch 印出訓練與驗證結果，方便觀察收斂情況
        print(
            f"Epoch {ep:03d} | "
            f"Train Loss={tl:.4f}, Acc={ta:.4f} | "
            f"Val Loss={vl:.4f}, Acc={va:.4f}"
        )

    # ---------- OUTPUT ----------
    # 建立輸出資料夾
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # 繪製 Accuracy 曲線
    plt.plot(history["train_acc"], label="Train accuracy")
    plt.plot(history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model A_Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "output_accuracy.png"))
    plt.close()

    # 繪製 Loss 曲線
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model A_Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "output_loss.png"))
    plt.close()

    # 將最終訓練結果輸出成 JSON
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
    # 使用訓練完成的模型對測試集進行預測
    model.eval()
    preds = []
    with torch.no_grad():
        for x in DataLoader(test_dataset, batch_size=64):
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())

    # 將預測的索引轉回原始標籤
    preds = [inv_map[p] for p in preds]

    # 將測試集預測結果輸出成 JSON
    with open(os.path.join(out_dir, "test_set_prediction.json"), "w", encoding="utf-8") as f:
        json.dump({"Predictions": preds}, f, indent=4)

    print("Model A done. All outputs generated.")
