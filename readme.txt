# CNN vs Residual CNN Experiment

## Project Overview（專案簡介）

This project implements and compares three CNN-based models under the same experimental setting.

本專案在相同訓練條件下，比較三種 CNN 模型的表現差異。

The goal is to analyze the effect of Residual Blocks and regularization techniques on model performance.

目的在於分析 Residual Block 與正則化技術對模型效能的影響。

---

## Project Navigation（專案導覽）

- [Baseline Model](Baseline/)
- [Model A](Model%20A/)
- [Model B](Model%20B/)

---

## Model Descriptions（模型說明）

### Baseline Model
Standard CNN without residual connections or regularization.

---

### Model A
CNN with Residual Blocks to improve training stability and gradient flow.

---

### Model B
Residual CNN with Data Augmentation and Dropout for better generalization.

---

## Results Visualization（結果視覺化）

### Baseline
![Baseline Loss](Baseline/output/output_loss.png)
![Baseline Accuracy](Baseline/output/output_accuracy.png)

---

### Model A
![Model A Loss](Model%20A/output/output_loss.png)
![Model A Accuracy](Model%20A/output/output_accuracy.png)

---

### Model B
![Model B Loss](Model%20B/output/output_loss.png)
![Model B Accuracy](Model%20B/output/output_accuracy.png)

---

## Output Files（輸出檔案）

Each model generates:

output/
├── output_loss.png
├── output_accuracy.png
├── output_output.json


Model B additionally outputs:

test_set_prediction.json


---

## Documents（文件）

- [Experiment Report](report.pdf)
- [Legacy Readme](readme.txt)

---

## Conclusion（結論）

- Baseline performs worst in convergence and accuracy  
- Residual connection improves training stability  
- Model B achieves best generalization performance  

---

## Repository Link

https://github.com/LeoChenJunan/cnn-residual-comparison
