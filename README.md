# M.E Final Year Project – Autonomous Driving using KITTI Dataset and Generative AI

## 📌 Project Overview

This project leverages the **KITTI Dataset** to explore and develop advanced computer vision techniques for autonomous driving. It utilizes both real and synthetically generated data using **GANs**, **VAEs**, and **LLMs** to train and test models for tasks such as object detection, stereo vision, and 3D object tracking.

---

## 📚 Dataset

### Name:
**KITTI Dataset**

### Description:
The KITTI Dataset, collected by the Karlsruhe Institute of Technology (KIT), is a comprehensive benchmark dataset for real-world driving scenarios. It includes:
- High-resolution stereo images
- 3D laser scans
- GPS/IMU data

It is categorized into subsets for:
- Stereo vision (disparity maps)
- Optical flow
- Object detection
- 3D object tracking

The data helps improve the robustness and generalizability of autonomous driving systems. The dataset continues to expand with new challenging scenarios like night driving and adverse weather conditions.

### Dataset URL:
- [KITTI on Kaggle](https://www.kaggle.com/datasets/klemenko/kitti-dataset)
- [GitHub Source](https://github.com/Arvind007-source/M.E-Final-Year-Project.git)

---

## 💻 Software Requirements

| Component           | Requirement                   |
|---------------------|-------------------------------|
| Operating System    | Windows 10/11, Ubuntu 20.04+, macOS |
| Python Version      | 3.8 or above                  |
| Libraries/Packages  | `numpy`, `pandas`, `matplotlib`, `tensorflow`/`pytorch`, `scikit-learn`, `opencv-python`, `seaborn`, `jupyter` |

---

## 🧠 Hardware Requirements

| Component              | Description                              |
|------------------------|------------------------------------------|
| GPU                    | Multi-GPU setup (e.g., NVIDIA A100)      |
| Storage                | High-Speed SSD (1TB+)                    |
| RAM                    | 32 GB or more                            |
| Computing Framework    | TensorFlow / PyTorch                     |

---

## 🤖 Generative AI & LLMs

| Category         | Models Used                                     |
|------------------|--------------------------------------------------|
| Generative Models| GAN, VAE, StyleGAN, CGAN, Beta-VAE              |
| Language Models  | GPT-2 (Used for synthetic scenario descriptions)|

---

## 🗂️ Data Sources

| Data Type        | Description                                      |
|------------------|--------------------------------------------------|
| Primary Dataset  | KITTI Dataset – Left/Right Stereo Images         |
| Augmented Data   | Synthetic images generated using GAN/VAE models  |

---

## 📏 Evaluation Metrics

| Metric             | Description                                     |
|--------------------|-------------------------------------------------|
| FID (Fréchet Inception Distance) | Measures synthetic image quality   |
| ROUGE              | Used to evaluate GPT-2 scenario outputs         |
| Perplexity         | Evaluates text generation coherence             |
| Semantic Similarity| Measures alignment between real & synthetic data|

---

## 🚗 Applications

- Autonomous Vehicle (AV) Training
- Simulation and Data Augmentation
- Safety and Scenario Testing

---

## ✅ How to Run the Project

### Step 1: Clone the Project Repository
```bash
git clone https://github.com/Arvind007-source/M.E-Final-Year-Project.git
```
### Step 2: Set Up Environment
```bash
pip install -r requirements.txt
```



