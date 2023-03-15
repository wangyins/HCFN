# HCFN (Pytorch Implementation)

EvalSet-1            |  EvalSet-2
:-------------------------:|:-------------------------:
![](https://github.com/wangyins/HCFN/blob/main/sample_videos/outputs/case1.gif)  |  ![](https://github.com/wangyins/HCFN/blob/main/sample_videos/outputs/case2.gif)
![](https://github.com/wangyins/HCFN/blob/main/sample_videos/outputs/case3.gif)  |  ![](https://github.com/wangyins/HCFN/blob/main/sample_videos/outputs/case4.gif)

This is an offical implementation for "HCFN: A Hierarchical Color Fusion Network for Exemplar-based Video Colorization".
## Table of Contents

- [Prerequisites](#Prerequisites)
- [Getting Started](#Getting-Started)

## Prerequisites
- Ubuntu 16.04
- Python 3.6.10
- CPU or NVIDIA GPU + CUDA 10.2 CuDNN
- PyTorch 1.5.1

## Getting Started

### Installation
- Clone this repository and install virtual environment.
```bash
git clone https://github.com/wangyins/HCFN
cd HCFN
conda env create -f environment.yaml
```
- Download model weights from <a href="https://drive.google.com/file/d/1r2SY8j8lzuN7vyzYjlehSy-Tsw05RKdl/view?usp=sharing">this link</a> to get "checkpoints.zip"
```bash
mkdir -p checkpoints/imagenet/
cd checkpoints/imagenet/
unzip checkpoints.zip
```
### Testing
```bash
sh test.sh
```
### Training
```bash
sh train.sh
```
