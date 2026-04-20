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

Evaluated inside **CARLA Simulator** — 10/30/50 vehicles, 10/30 walkers across weather conditions.

### Inference Time: ViT vs CNN

| Weather Condition | Frames Tested | ViT Avg (ms) | CNN Avg (ms) | ViT FPS | CNN FPS | CNN Speedup |
|-------------------|--------------|-------------|-------------|---------|---------|-------------|
| ClearNoon         | 300          | 673.0 ms    | 93.7 ms     | 1.49    | 10.68   | **7.2x faster** |
| CloudyNoon        | 36           | 747.2 ms    | 106.5 ms    | 1.34    | 9.39    | **7.0x faster** |

### Key Findings

- CNN is **~7x faster** than ViT across all tested conditions
- ViT inference degrades further under **CloudyNoon** (+74ms vs ClearNoon)
- CNN remains **stable** across weather conditions (~13ms variance)
- CNN is significantly more suitable for **real-time autonomous deployment**
- ViT's higher computational cost makes it challenging for **edge devices**

> Full frame-by-frame benchmark data available in `benchmark_results.xlsx`
### Key Findings

- CNN is approximately **7.7x faster** than ViT in real-time inference
- ViT inference time ranges from **623ms to 682ms** per frame
- CNN inference time ranges from **76ms to 92ms** per frame
- CNN is significantly more suitable for **real-time edge deployment**
- ViT may be preferred where **accuracy over speed** is the priority

> Full frame-by-frame data available in `benchmark_results.xlsx`


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
## Setup & Requirements

```bash
Python >= 3.8
PyTorch >= 1.12
timm
opencv-python
numpy
matplotlib
carla
```

```bash
pip install -r requirements.txt
```

---

## Usage

**1. Launch CARLA server**
```bash
./CarlaUE4.sh -quality-level=Epic -fps=30
```

**2. Collect data**
```bash
python _data_collection/data_collector.py
```

**3. Run comparison**
```bash
python Autonomous_VIT_CNN.py
```

**4. View benchmark results**
```bash
python thesis_test.py
```

---

## Applications

Findings from this benchmark are applicable to:
- **Self-driving vehicles** — camera-based perception modules
- **Mobile robots** — navigation and obstacle avoidance on edge hardware
- **Embedded AI systems** — architecture selection under memory and
  compute constraints

---

## Author

**Sagar Allu**
Master Thesis Project — [University of Europe for Applied Sciences]

---

## Citation

```bibtex
@thesis{allu2025cnnvit,
  title  = {A Comparison of Vision Transformer and CNN for Edge Devices},
  author = {Sagar Allu},{Dr.Aditya Mushyam}
  school = {University of Europe for Applied sciences,Potsdam},
  year   = {2025}
}
```

---

## License

This repository is released for academic and research use only.
