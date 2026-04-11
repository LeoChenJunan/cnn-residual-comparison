# CNN vs Residual CNN Experiment

## 一、模型架構設計說明

本實驗設計三種卷積神經網路（Convolutional Neural Network, CNN）模型，目的在於系統性分析 Residual Block 以及正則化技術（Data Augmentation 與 Dropout）對模型訓練行為與效能的影響。

為確保實驗比較的公平性，三個模型皆採用相同的輸入大小與基礎架構設定。所有模型的輸入影像大小皆為 3 × 32 × 32，前段皆使用一層卷積層將輸入影像轉換為特徵圖；中段則堆疊多個卷積區塊（Block），僅在 Block 類型上有所差異；後段分類器皆採用 Global Average Pooling（GAP），以降低參數量並避免過擬合。

---

### （一）Baseline 模型（Plain CNN）

Baseline 模型使用 Plain Block 作為基本建構單元，其結構為兩層 3×3 卷積層，每層卷積後接一個 ReLU 非線性函數，但不包含 Batch Normalization 與 Skip Connection。

此模型作為傳統 CNN 的對照組，用以觀察在相同深度下，缺乏現代深度學習機制時的訓練與泛化表現。

---

### （二）Model A（Residual CNN）

Model A 在架構上與 Baseline 相同，但將中段的 Plain Block 改為 Residual Block。

每個 Residual Block 由兩層「Convolution + Batch Normalization」所組成，並透過 Identity Shortcut 將輸入直接與主路徑輸出進行相加，最後再經由 ReLU 激活函數。

此設計可在不增加額外參數的情況下，改善深層網路的梯度傳遞問題，有助於模型在訓練過程中更穩定收斂。

---

### （三）Model B（Residual CNN + Regularization）

Model B 的整體架構與 Model A 完全相同，差異僅在於訓練策略與分類器設計。

在訓練階段加入 Data Augmentation（如隨機翻轉與裁切），以增加資料多樣性；並於分類器中額外加入 Dropout（p = 0.5），作為正則化手段，以降低模型過擬合的風險。

---

## 二、實驗設計與比較說明

本實驗在相同訓練設定下進行三組模型比較，以確保結果具有可比性：

- Baseline：無 Residual Block，無 Data Augmentation 與 Dropout  
- Model A：加入 Residual Block，不使用正則化技術  
- Model B：在 Model A 基礎上，加入 Data Augmentation 與 Dropout  

三組模型皆使用以下相同訓練參數：

- Learning Rate：1e-3  
- Batch Size：64  
- Epoch 數：50  

---

## 三、實驗結果分析

由實驗結果可觀察到，Baseline 模型在訓練與驗證階段的表現明顯落後於 Model A 與 Model B。其收斂速度較慢，且驗證準確率提升有限，顯示在相同網路深度下，僅透過單純堆疊卷積層，模型容易面臨訓練困難與泛化能力不足的問題。

Model A 在引入 Residual Block 後，訓練過程較為穩定，Loss 下降速度加快，且驗證準確率相較 Baseline 有顯著提升。此結果驗證 Residual Connection 能有效改善深層網路的可訓練性，使模型更容易學習有效特徵。

Model B 則在 Model A 基礎上進一步加入正則化技術。雖然訓練準確率略低於 Model A，但在驗證集上的表現最佳，且驗證 Loss 未出現明顯反彈，顯示 Data Augmentation 與 Dropout 能有效抑制過擬合並提升泛化能力。

---

## 四、心得與討論

透過本次實驗可以發現，Residual Block 不僅能提升模型效能，更重要的是改善深層網路的訓練穩定性，使模型在相同深度下更容易收斂。

在此基礎上加入 Data Augmentation 與 Dropout，雖然可能略微犧牲訓練集表現，但能換取更穩定且可靠的驗證結果，對實際應用而言更具價值。

相較於單純追求高訓練準確率，Model B 的設計更符合現代深度學習對模型泛化能力的重視，也更貼近實務中的模型設計思維。

此次實驗讓我更清楚理解各種網路設計與正則化技術在實際訓練過程中所扮演的角色，也讓我體會到模型設計不僅是堆疊層數，更需要考慮訓練穩定性與實際應用情境。
