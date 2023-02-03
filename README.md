# HCFN
This is an offical implementation for [HCFN: A Hierarchical Color Fusion Network for Exemplar-based Video Colorization], and the current code works well with PyTorch 1.5.1.
## Table of Contents

- [Prerequisites](#Prerequisites)
- [Getting Started](#Getting-Started)
- [Citation](#Citation)

## Prerequisites
- Ubuntu 16.04
- Python 3.6.10
- CPU or NVIDIA GPU + CUDA 10.2 CuDNN

## Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/wangyins/HCFN
cd HCFN
conda env create -f environment.yaml
```
- Download model weights from <a href="https://drive.google.com/file/d/1r2SY8j8lzuN7vyzYjlehSy-Tsw05RKdl/view?usp=sharing">[Google Drive]</a> to get "checkpoints.zip"
```bash
mkdir -p checkpoints/imagenet/
cd checkpoints/imagenet/
unzip checkpoints.zip
```
### Testing
```bash
sh test.sh
```
## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{yin_2023,
  title={HCFN: A Hierarchical Color Fusion Network for Exemplar-based Video Colorization},
  author={Yin, Wang and Lu, Peng and Peng, XuJun and Zhao, ZhaoRan and Yu, JinBei},
  booktitle={},
  year={2023}
}
```
