# CNN vs Residual CNN Experiment

## Project Overview

This project compares three convolutional neural network (CNN) architectures under the same training setting to analyze the impact of Residual Blocks and regularization techniques on model performance and generalization ability.

The three models include:
- Baseline CNN (no Residual Block, no regularization)
- Residual CNN (Model A)
- Residual CNN with Data Augmentation and Dropout (Model B)

All models are trained on images with input size 3 × 32 × 32 and evaluated under identical training conditions.

---

## Model Architectures

### Baseline Model
The Baseline model is a standard CNN composed of stacked convolutional layers with ReLU activation. It does not include Batch Normalization, Residual Connections, or any form of regularization. This model serves as the control group for comparison.

---

### Model A (Residual CNN)
Model A introduces Residual Blocks into the network architecture. Each Residual Block consists of two convolutional layers with Batch Normalization and an identity shortcut connection. This design improves gradient flow and stabilizes training in deeper networks.

---

### Model B (Residual CNN + Regularization)
Model B is based on Model A but further incorporates regularization techniques. Data Augmentation (random flipping and cropping) is applied during training, and Dropout (p = 0.5) is added in the classifier to reduce overfitting and improve generalization performance.

---

## Training Configuration

All models are trained using the same hyperparameters:

- Learning Rate: 1e-3  
- Batch Size: 64  
- Epochs: 50  

---

## Experimental Results

### Accuracy Comparison

<p align="center">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_accuracy.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_accuracy.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_accuracy.png" width="32%">
</p>

---

### Loss Comparison

<p align="center">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Baseline/output/output_loss.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_A/output/output_loss.png" width="32%">
  <img src="https://raw.githubusercontent.com/LeoChenJunan/cnn-residual-comparison/main/Model_B/output/output_loss.png" width="32%">
</p>

---

## Results Analysis

The Baseline model shows the weakest performance in both convergence speed and validation accuracy, indicating limitations in deep feature learning without modern architectural improvements.

Model A demonstrates improved training stability and faster convergence due to the use of Residual Connections, which help alleviate gradient vanishing problems in deeper networks.

Model B achieves the best generalization performance. Although its training accuracy is slightly lower than Model A, it produces the most stable validation results, confirming the effectiveness of Data Augmentation and Dropout in reducing overfitting.

---

## Project Structure

The repository is organized as follows:

- Baseline/
- Model_A/
- Model_B/
- readme.txt
- report.pdf

Each model folder contains:
- main.py (training script)
- output/ (loss curve, accuracy curve, prediction results)

---

## Conclusion

This experiment demonstrates that architectural improvements such as Residual Connections significantly enhance training stability and convergence behavior in CNNs. Furthermore, applying regularization techniques such as Data Augmentation and Dropout improves model generalization, making the model more robust in practical scenarios.

Overall, Model B achieves the best balance between training performance and generalization ability.

---

## Repository

https://github.com/LeoChenJunan/cnn-residual-comparison
