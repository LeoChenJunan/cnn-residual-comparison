# CNN vs Residual CNN Experiment

---

## 一、模型架構設計說明

本實驗設計三種卷積神經網路（CNN）模型，目的在於系統性分析 Residual Block 以及正則化技術（Data Augmentation 與 Dropout）對模型訓練行為與效能的影響。

為確保實驗公平性，三個模型皆使用相同輸入大小 3 × 32 × 32，前段為卷積層，中段為不同 Block，後段皆使用 Global Average Pooling（GAP）。

---

### Baseline
Baseline：無 Residual Block，無 Data Augmentation 與 Dropout  

---

### Model A
Model A：加入 Residual Block，不使用正則化技術  

---

### Model B
Model B：在 Model A 基礎上，加入 Data Augmentation 與 Dropout  

---

## 二、模型比較（Accuracy Visualization）

### Accuracy Comparison (Baseline vs Model A vs Model B)

<p align="center">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_accuracy.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_accuracy.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_accuracy.png" width="32%">
</p>

---

## 三、Loss 比較

<p align="center">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_loss.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_loss.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_loss.png" width="32%">
</p>

---

## 四、實驗結果分析

Baseline 收斂較慢且準確率最低。  
Model A 因 Residual Block 提升梯度傳遞，使訓練更穩定。  
Model B 加入 Data Augmentation 與 Dropout，泛化能力最佳。

---

## 五、實驗設計

- Learning Rate：1e-3  
- Batch Size：64  
- Epoch：50  

---

## 六、專案結構

- Baseline/
- Model_A/
- Model_B/
- readme.txt
- report.pdf

---

## 七、Repository

👉 https://github.com/LeoChenJunan/cnn-residual-comparison
