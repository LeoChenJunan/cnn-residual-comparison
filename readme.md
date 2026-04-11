# CNN vs Residual CNN Experiment

## Project Overview（專案簡介）

This project implements and compares three CNN-based models under the same experimental setting.

本專案在相同訓練條件下，比較三種 CNN 模型的表現差異。

The goal is to analyze how Residual Blocks and regularization techniques affect model performance and generalization ability.

目的在於分析 Residual Block 與正則化技術對模型效能與泛化能力的影響。

---

## Project Structure（專案結構）

The repository contains three model folders:

本專案包含三個模型資料夾：

- [Baseline Model](Baseline/)
- [Model A](Model_A/)
- [Model B](Model_B/)

Each folder includes:
- main.py (training script)
- train.json (not uploaded due to size limitation)
- output/ (training results)

---

## Model Description（模型說明）

Baseline: standard CNN without residual connections or regularization.

Baseline：一般 CNN，無 Residual 與正則化。

Model A: CNN with Residual Blocks to improve gradient flow and training stability.

Model A：加入 Residual Block 提升梯度傳遞與訓練穩定性。

Model B: Residual CNN with Data Augmentation and Dropout to improve generalization.

Model B：在 Model A 基礎上加入 Data Augmentation 與 Dropout 提升泛化能力。

---

## Results（實驗結果）

### Baseline
![Baseline Loss](https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_loss.png)

![Baseline Accuracy](https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_accuracy.png)

---

### Model A
![Model A Loss](https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_loss.png)

![Model A Accuracy](https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_accuracy.png)

---

### Model B
![Model B Loss](https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_loss.png)

![Model B Accuracy](https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_accuracy.png)

---

## Output Files（輸出檔案）

Each model generates:

output/
├── output_loss.png
├── output_accuracy.png
├── output_output.json


Model B additionally produces:

test_set_prediction.json


---

## Documents（文件）

- [Experiment Report](report.pdf)
- [Legacy Readme](readme.txt)

---

## Conclusion（結論）

- Baseline shows weakest performance
- Residual connection improves training stability and convergence
- Model B achieves best generalization performance

---

## Repository

https://github.com/LeoChenJunan/cnn-residual-comparison
