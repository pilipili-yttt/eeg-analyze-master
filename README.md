# eeg-analyze-master


## 📌 项目简介

本项目旨在对EEG（脑电）信号的片段进行特征提取与回归预测，结合了功率谱密度（PSD）与相位锁定值（PLV）等频域特征，采用图神经网络（GCN）与卷积神经网络（CNN）进行建模与训练，实现精确的目标变量回归。

---

## 🧩 功能模块

### 1. 特征提取模块（`FeatureExtractor`）

- 对输入的两组EEG数据进行预处理、降采样与统一长度处理。
- 计算每个通道的PSD特征，划分为多个频带（delta、theta、alpha、beta、low-gamma、high-gamma）。
- 计算所有通道对之间的PLV相位一致性矩阵。
- 支持特征缓存避免重复计算。

### 2. 神经网络回归模块（`Model` + `RegressionOpti`）

- 使用GCN处理PSD频带特征的图结构表示。
- 使用CNN提取PLV矩阵中的局部特征。
- 融合两种特征后通过多层全连接网络进行回归预测。
- 提供并发训练支持，多线程模型池、训练缓存与早停机制。

---



## ⚙️ 安装与依赖

### 📦 Python 依赖

 Python 3.8+：

```bash
pip install numpy scipy tqdm torch torch_geometric
```

---



## 🧠 主要类与方法说明

### `FeatureExtractor(p_1, p_2, ...)`
用于从两组EEG数据中提取PSD、PLV与对应标签。

- `get_PSD(channels)`：获取指定通道的频带能量。
- `get_PLV(channels)`：获取通道对应的PLV矩阵。
- `get_Y()`：返回归一化后的标签（基于片段时长）。

---

### `Model`
集成 GCN、CNN 和全连接层，用于处理图结构的频带特征与图像结构的相干矩阵。

---

### `RegressionOpti`
管理多线程训练图神经网络，支持自动验证、早停与缓存机制。

- `train_eval(data)`：训练并返回最终预测误差。

---

## 📁 文件结构

```
.
│  feature_extraction.py     # 提取EEG信号的时频域特征
│  GA_optimize.py            # 使用遗传算法优化通道选择
│  ica_comparison.py         # 比较不同ICA方法的信号分离效果
│  idx2chName.py             # 通道索引与通道名称之间的映射工具
│  NN_train.py               # 神经网络模型训练代码
│  README.md                 # 本文件
│  test.py                   # 快速测试脚本入口
│
└─preprocess/
        neuroscan.py         # 处理Neuroscan格式的EEG数据，如读取.cnt文件、滤波、去伪迹等
```

---

