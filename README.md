# A Comparison of Vision Transformer and CNN for Edge Devices
> Benchmarking ViT vs CNN for autonomous systems using the CARLA simulator

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-0.9.13-blueviolet)
![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx&logoColor=white)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Integrated-00FFAA)
![License](https://img.shields.io/badge/License-Academic-green)
---

## Overview

This project presents a systematic comparison between **Convolutional Neural
Networks (CNN)** and **Vision Transformers (ViT)** for visual perception tasks
targeting edge deployment in autonomous systems.

Using the **CARLA autonomous driving simulator**, both architectures are
evaluated across parameters critical to real-world deployment — including
inference speed, parameter efficiency, accuracy, and memory footprint —
on hardware-constrained edge devices such as those found in self-driving
vehicles and robotic platforms.

---

## Why This Comparison Matters

CNNs have been the backbone of computer vision for years, offering strong
inductive biases and efficient inference. Vision Transformers, however, bring
global context reasoning via self-attention — a potentially powerful advantage
for scene understanding in complex driving environments.

This work answers: **Which architecture is more suitable for edge-deployed
autonomous perception — and under what conditions?**

---

## Architectures Compared

| Feature                    | CNN                        | Vision Transformer (ViT)       |
|----------------------------|----------------------------|--------------------------------|
| Core mechanism             | Convolutional layers       | Self-attention on image patches |
| Receptive field            | Local → gradually global   | Global from the first layer    |
| Inductive bias             | Strong (locality, translation equivariance) | Weak (data-hungry) |
| Inference latency          | Generally lower            | Higher (quadratic complexity)  |
| Edge suitability           | Proven                     | Improving (pruning/quantization) |
| Robustness to occlusion    | Moderate                   | Stronger (global context)      |

---

## Benchmark Results

All models were evaluated inside the CARLA simulator under identical conditions.

| Model        | Accuracy (%) | Inference Time (ms) | FPS  | Parameters  | Memory (MB) |
|--------------|-------------|---------------------|------|-------------|-------------|
| CNN          | xx.x        | xx.x                | xx   | xxM         | xxx         |
| Faster CNN   | xx.x        | xx.x                | xx   | xxM         | xxx         |
| SSD CNN      | xx.x        | xx.x                | xx   | xxM         | xxx         |
| YOLO CNN     | xx.x        | xx.x                | xx   | xxM         | xxx         |
| ViT          | xx.x        | xx.x                | xx   | xxM         | xxx         |
| ViT+CNN      | xx.x        | xx.x                | xx   | xxM         | xxx         |

> Full benchmark data available in `benchmark_results.xlsx`
---

## Simulator: CARLA

All data collection and evaluation is performed in **CARLA** (Car Learning to
Act), an open-source simulator for autonomous driving research.

CARLA provides:
- Photorealistic urban and rural environments
- Configurable weather: rain, fog, night, direct sunlight
- Sensor suite: RGB cameras, depth maps, semantic segmentation, LiDAR
- Scriptable traffic, pedestrians, and dynamic obstacles

---
## Simulation Screenshots

### CARLA Environment
![Simulation](recorded_images/your_screenshot.png)

### Detection Output
![Detection](recorded_images/your_detection.png)


## Project Structure

```
Autonomous vehicle/
├── agents/                          # CARLA agent scripts
├── _data_collection/                # Raw data from CARLA
├── recorded_data/                   # Recorded sensor data
├── recorded_images/                 # Captured frames
├── manual_data/                     # Manual driving data
├── PythonAPI/                       # CARLA Python API
├── yolov5-master/                   # YOLOv5 integration
│
├── Autonomous_VIT_CNN.py            # Main comparison script
├── Autonomus.py                     # Core autonomous driving
├── faster_cnn.py                    # Faster CNN implementation
├── Yolo_CNN.py                      # YOLO + CNN hybrid
├── thesis_test.py                   # Thesis evaluation
│
├── cnn_model.onnx                   # Exported CNN model
├── vit_model.onnx                   # Exported ViT model
│
├── benchmark_results.xlsx           # Benchmark data
├── performance_log.xlsx             # Performance metrics
└── main.py                          # Entry point
```
