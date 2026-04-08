## 模型設計

### Baseline

使用 Softmax Regression 作為最基本模型
將影像攤平成向量後進行分類
不包含任何卷積結構或進階技巧

### Model A

使用卷積神經網路並加入 Residual Block
透過 skip connection 改善梯度傳遞問題
提升模型訓練穩定性與收斂速度

### Model B

在 Model A 基礎上加入正則化方法
包含資料增強與 dropout
用於提升模型泛化能力並減少過擬合

---

## 實驗設定

* Learning rate：1e-3
* Batch size：64
* Epoch：50
* Train validation split：80 20

---

## 實驗結果

* Baseline 表現最差且收斂較慢
* Model A 提升訓練穩定性與準確率
* Model B 在驗證集表現最佳，泛化能力較佳

---

## 專案重點

* 實作 CNN 與 Residual Block
* 理解模型設計對訓練的影響
* 比較不同正則化方法效果
* 建立完整訓練與評估流程

---

## 使用技術

* Python
* NumPy
* PyTorch
* Matplotlib

---

資料集說明:
本專案使用之資料集因檔案較大未上傳至GitHub 以避免超過平台限制 專案主要提供模型架構與訓練流程 如需完整資料下載方式可聯繫信箱 "leochenjunan@gmail.com"


---

