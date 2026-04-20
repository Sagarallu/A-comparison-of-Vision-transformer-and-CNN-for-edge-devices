# A Comparison of Vision Transformer and CNN for Edge Devices
> Benchmarking ViT vs CNN for autonomous systems using the CARLA simulator

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange)
![CARLA](https://img.shields.io/badge/CARLA-0.9.x-purple)
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

## Evaluation Metrics

The following parameters are benchmarked inside the CARLA simulation environment:

- **Inference latency** (ms per frame)
- **Frames per second (FPS)** — real-time capability
- **Model parameter count**
- **Memory footprint** (RAM / GPU VRAM)
- **Top-1 Accuracy / mAP** on CARLA-captured scenes
- **Robustness** under weather and lighting variations

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

## Project Structure
