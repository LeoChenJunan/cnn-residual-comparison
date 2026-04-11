# CNN vs Residual CNN 實驗比較

---

## 一、專案簡介

本專案透過三種卷積神經網路（CNN）架構進行比較實驗，分析 Residual Block（殘差連接）以及正則化技術（Data Augmentation 與 Dropout）對模型訓練穩定性與泛化能力的影響。

本實驗設計三種模型：

- Baseline（基準模型）
- Model A（加入 Residual Block）
- Model B（Residual + 正則化）

所有模型皆在相同資料集與訓練設定下進行比較，以確保實驗公平性。

---

## 二、模型架構說明

### （一）Baseline 模型

Baseline 為標準卷積神經網路，由多層卷積層與 ReLU 激活函數組成，不包含 Batch Normalization、Residual Connection 或任何正則化方法。

此模型用來作為「最基本深度學習架構」的比較基準。

---

### （二）Model A（Residual CNN）

Model A 在架構中加入 Residual Block（殘差結構），透過 shortcut connection 將輸入直接加到輸出上。

此設計帶來以下改善：

- 改善梯度消失問題  
- 加快收斂速度  
- 提升深層網路訓練穩定性  

因此 Model A 通常能比 Baseline 更快達到較高準確率。

---

### （三）Model B（Residual + 正則化）

Model B 在 Model A 基礎上加入兩種正則化策略：

- Data Augmentation（隨機翻轉、裁切）
- Dropout（p = 0.5）

其目的為：

- 增加資料多樣性  
- 降低模型過擬合風險  
- 提升泛化能力（validation performance）  

---

## 三、訓練設定

所有模型皆使用相同設定：

- Learning Rate：1e-3  
- Batch Size：64  
- Epoch：50  
- 輸入大小：3 × 32 × 32  

---

## 四、實驗結果對比（重點分析）

### （1）整體趨勢比較

從三個模型的訓練結果可以觀察到明顯的性能階層：

> Baseline < Model A < Model B（泛化最佳）

---

### （2）Baseline 表現分析

Baseline 模型表現最弱，主要原因如下：

- 無 Residual 結構，深層訓練時梯度傳遞效率較差  
- 容易出現收斂速度慢的問題  
- validation accuracy 提升有限  
- 模型容易陷入局部最優解  

👉 結論：  
Baseline 顯示「純堆疊卷積層」在較深網路中存在明顯限制。

---

### （3）Model A（Residual CNN）改善分析

Model A 相較 Baseline 有明顯提升：

- Loss 收斂速度更快  
- 訓練過程更穩定（震盪減少）  
- validation accuracy 明顯提升  
- 深層特徵學習能力更好  

👉 核心原因：

Residual connection 提供「捷徑路徑」，讓梯度可以直接回傳，減少梯度消失問題。

👉 結論：  
Residual Block 是提升 CNN 深層訓練穩定性的關鍵技術。

---

### （4）Model B（Residual + 正則化）最佳表現分析

Model B 在訓練集表現可能略低於 Model A，但在驗證集表現最佳。

具體觀察：

- training accuracy 略下降（正常現象）  
- validation accuracy 最高  
- validation loss 最穩定  
- overfitting 明顯減少  

👉 原因分析：

1. Data Augmentation 增加資料變化，使模型不依賴單一特徵  
2. Dropout 降低神經元共適應（co-adaptation）  
3. 模型泛化能力提升，因此在未知資料表現更好  

👉 結論：

Model B 雖然「訓練表現不是最高」，但它是**實際應用中最有價值的模型**。

---

### （5）整體比較結論

三者差異可以總結為：

- Baseline：學得慢、效果有限（基準模型）
- Model A：學得快、穩定性提升（架構改良）
- Model B：泛化最佳、最接近實務應用（完整優化）

---

## 五、實驗結果圖

### Accuracy 比較

<p align="center">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_accuracy.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_accuracy.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_accuracy.png" width="32%">
</p>

---

### Loss 比較

<p align="center">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_loss.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_loss.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_loss.png" width="32%">
</p>

---

## 六、專案結構

- Baseline/
- Model_A/
- Model_B/
- readme.txt
- report.pdf

---

## 七、結論

本實驗驗證了三個重要觀察：

1. 單純堆疊 CNN 在深層結構中效果有限  
2. Residual Connection 能顯著提升訓練穩定性與收斂速度  
3. 正則化技術能有效提升模型泛化能力，使模型更適合實務應用  

整體而言，Model B 為最佳模型，在訓練穩定性與泛化能力之間取得最佳平衡。

---

## 八、Repository

https://github.com/LeoChenJunan/cnn-residual-comparison
