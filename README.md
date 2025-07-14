# Ultrafast Phase-Transition Temporal-Coding Neurons for Energy-Efficient Federated Neuromorphic Computing

This repository provides the implementation of **federated neuromorphic learning framework** based on  temporal-coding neurons. The framework is designed for energy-efficient and real-time processing, suitable for both single-modal and multimodal edge intelligence tasks.

## Overview

We propose a spike-based neuromorphic computing system leveraging TC neurons . To demonstrate its effectiveness, we present a **vehicle cognition task** that integrates both image and audio modalities under a **federated learning** scheme.

## Project Structure
The project consists of two main parts: the computing framework (`SNN-simulator`) and a set of illustrative examples.

A vehicle cognition task is provided as a representative use case, where the computing framework is employed as a Python module.

## Quick start
### 1. Insatll dependencies
See requirements.txt for a list of required Python packages.

### 2. Run examples
* img-main.py: Train and test the image modal net on the provided image dataset.
* sound-main.py: Train and test the sound modal net on the provided sound dataset.
* multimodal.py: Train and test the multimodal net on the provided multimodal dataset.
* fl-fuse-main.py: Train and test the multimodal net in federated learning method.
