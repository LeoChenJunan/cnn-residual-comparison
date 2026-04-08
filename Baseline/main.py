"""
Baseline: Plain Softmax Regression (No Residual, No Augmentation)
此模型作為本次實驗的對照組，使用最基本的 Softmax Regression，
不包含 Residual Block、Data Augmentation 或 Dropout。
"""

import os                       # 匯入 os，用來處理資料夾建立與路徑操作
import json                     # 匯入 json，負責讀取訓練資料與輸出結果檔
import numpy as np              # 匯入 numpy，進行向量與矩陣運算
import matplotlib.pyplot as plt # 匯入 matplotlib，用來繪製 Loss / Accuracy 曲線

# ---------------- Reproducibility ----------------
np.random.seed(42)  
# 固定亂數種子，確保每次訓練結果一致，方便實驗比較與重現結果

# ---------------- JSON LOADING ----------------
def load_train_json(path):
    # 開啟訓練資料的 json 檔案
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 將 Image 欄位取出並轉成 numpy array
    X = np.array([d["Image"] for d in raw], dtype=np.float32)
    
    # 將 Label 欄位取出，作為分類的真實標籤
    y = np.array([d["Label"] for d in raw], dtype=np.int64)

    # 將 3×32×32 的影像攤平成一維向量 (3072)
    # 這是 Softmax Regression 必須的輸入格式
    X = X.reshape(len(X), -1)  # (N, 3072)
    return X, y


def load_test_json(path):
    # 開啟測試資料 json（沒有標籤）
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 只讀取 Image 欄位
    X = np.array([d["Image"] for d in raw], dtype=np.float32)
    
    # 同樣攤平成一維向量，與訓練資料格式一致
    X = X.reshape(len(X), -1)
    return X


# ---------------- BASIC OPS ----------------
def softmax(z):
    # 為了數值穩定性，先減去每一列的最大值
    z = z - np.max(z, axis=1, keepdims=True)
    
    # 計算指數
    exp_z = np.exp(z)
    
    # 將每一列正規化成機率分佈
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, num_classes):
    # 建立 one-hot 編碼矩陣，初始值全為 0
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    
    # 將對應類別位置設為 1
    oh[np.arange(len(y)), y] = 1.0
    return oh


def evaluate(X, Y, W, B):
    # 計算模型輸出的類別機率
    probs = softmax(X @ W + B)
    
    # 使用 Cross Entropy 計算 Loss
    loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
    
    # 計算分類正確率
    acc = np.mean(np.argmax(probs, axis=1) == np.argmax(Y, axis=1))
    return loss, acc


# ---------------- TRAINING ----------------
def train_softmax(X_train, Y_train, X_val, Y_val, lr, epochs, batch_size):
    # 取得資料筆數與特徵維度
    n, d = X_train.shape
    
    # 類別數量
    k = Y_train.shape[1]

    # 初始化權重矩陣（小隨機值）
    W = np.random.randn(d, k) * 0.01
    
    # 初始化 bias 為 0
    B = np.zeros((1, k))

    # 用來記錄訓練與驗證過程中的 Loss 與 Accuracy
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Epoch 訓練迴圈
    for ep in range(1, epochs + 1):
        # 隨機打亂資料順序，避免學到資料排序的偏差
        indices = np.random.permutation(n)
        Xs, Ys = X_train[indices], Y_train[indices]

        # Mini-batch 訓練
        for i in range(0, n, batch_size):
            xb = Xs[i:i + batch_size]
            yb = Ys[i:i + batch_size]

            # 前向傳播，計算機率
            probs = softmax(xb @ W + B)
            
            # 計算梯度（Cross Entropy + Softmax 的標準結果）
            grad = (probs - yb) / len(xb)

            # 使用梯度下降更新權重與 bias
            W -= lr * xb.T @ grad
            B -= lr * np.sum(grad, axis=0, keepdims=True)

        # 每個 epoch 結束後評估整體訓練與驗證表現
        train_loss, train_acc = evaluate(X_train, Y_train, W, B)
        val_loss, val_acc = evaluate(X_val, Y_val, W, B)

        # 紀錄歷史數據（轉成 float 方便後續輸出 JSON）
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        # 印出每個 epoch 的訓練狀況
        print(
            f"Epoch {ep:03d} | "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
        )

    return W, B, history


# ---------------- MAIN ----------------
if __name__ == "__main__":

    # -------- Load data --------
    # 載入訓練資料與標籤
    X_all, y_all = load_train_json("train.json")
    
    # 載入測試資料（無標籤）
    X_test = load_test_json("test.json")

    # 將像素值正規化到 [0, 1]，有助於模型訓練穩定
    X_all /= 255.0
    X_test /= 255.0

    # -------- Train / Val split (80 / 20) --------
    # 計算總資料筆數
    num_samples = len(X_all)
    
    # 隨機打亂索引
    indices = np.random.permutation(num_samples)
    
    # 以 8:2 比例切分訓練與驗證資料
    split = int(0.8 * num_samples)

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # -------- Label mapping (ONLY from train set) --------
    # 只使用訓練集的 label 建立 mapping，避免資料洩漏
    labels = np.unique(y_train)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_map = {i: int(l) for l, i in label_map.items()}

    # 將原始 label 轉成連續索引
    y_train_idx = np.array([label_map[y] for y in y_train])
    y_val_idx = np.array([label_map[y] for y in y_val])

    # 轉成 one-hot 格式
    Y_train = one_hot(y_train_idx, len(labels))
    Y_val = one_hot(y_val_idx, len(labels))

    # -------- Training params --------
    # 設定訓練超參數
    params = {
        "lr": 0.01,
        "epochs": 50,
        "batch_size": 64
    }

    # -------- Train --------
    # 開始訓練 Softmax Regression 模型
    W, B, hist = train_softmax(
        X_train, Y_train,
        X_val, Y_val,
        lr=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )

    # 計算最終訓練與驗證結果
    final_train_loss, final_train_acc = evaluate(X_train, Y_train, W, B)
    final_val_loss, final_val_acc = evaluate(X_val, Y_val, W, B)

    # -------- Output --------
    # 建立 output 資料夾（若不存在）
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy 曲線
    plt.plot(hist["train_acc"], label="Train accuracy")
    plt.plot(hist["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline_Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "output_accuracy.png"))
    plt.close()

    # Loss 曲線
    plt.plot(hist["train_loss"], label="Train loss")
    plt.plot(hist["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline_Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "output_loss.png"))
    plt.close()

    # 將最終結果輸出成 JSON 檔
    with open(os.path.join(out_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump({
            "Learning rate": params["lr"],
            "Epoch": params["epochs"],
            "Batch size": params["batch_size"],
            "Final train accuracy": float(final_train_acc),
            "Validation accuracy": float(final_val_acc),
            "Final train loss": float(final_train_loss),
            "Final validation loss": float(final_val_loss)
        }, f, indent=4)

    # -------- Test prediction --------
    # 對測試集進行預測
    probs = softmax(X_test @ W + B)
    preds = [inv_map[int(i)] for i in np.argmax(probs, axis=1)]

    # 輸出測試集預測結果
    with open(os.path.join(out_dir, "test_set_prediction.json"), "w", encoding="utf-8") as f:
        json.dump({"Predictions": preds}, f)

    print("Baseline finished. Outputs generated.")
