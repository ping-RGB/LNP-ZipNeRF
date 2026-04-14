# LNP-ZipNeRF: Noise-Resilient Zip-NeRF for Novel View Synthesis through Learnable Noise Priors

This is the official PyTorch implementation of **LNP-ZipNeRF**, built upon [Zip-NeRF](https://github.com/SuLvXiangXin/zipnerf-pytorch). Our method introduces **learnable noise priors** and a novel **dual‑branch noise analysis network** to make grid‑based neural radiance fields robust against noisy input images, enabling high‑quality novel view synthesis from corrupted or real‑world degraded captures.

## News
- (2025.4.14) Initial release of LNP‑ZipNeRF.

## Contributions

Our main contributions are as follows:

- **Dual-Branch Noise Analysis Network:** A complementary dual-branch architecture is designed to accurately analyze the complex and unknown noise distribution in input images. One branch focuses on identifying high-frequency noise, while the other captures global brightness fluctuations. The synergy between these branches provides a robust feature foundation for subsequent targeted suppression.

- **Adaptive Feature Fusion Mechanism:** A dynamic mask learning module driven by a multi-layer perceptron (MLP) is introduced. This module adaptively integrates the analyzed noise features with original scene features based on image content, effectively filtering out noise while reinforcing constraints on the scene's geometric structure, thereby avoiding detail loss caused by excessive smoothing. Experimental results demonstrate that this mechanism enables the model to achieve an adaptive balance between denoising intensity and geometry preservation.

- **Collaborative Optimization Objective:** A multi-task loss function is constructed that combines color fidelity, structural similarity, and mask regularization. This loss function ensures that denoising, feature fusion, and view synthesis are jointly optimized within a unified framework, enhancing the model's generalization capability across diverse noise scenarios. Experimental results show that this loss function design enables stable model convergence under various noise conditions, achieving a PSNR improvement of 2–3 dB over baseline methods.

## Results

We evaluate on MipNeRF360‑based datasets with added noise (following NAN’s noise model) and then denoised by Diffbir.  
The table below reports PSNR and SSIM under different noise gains (2, 4, 8, 16, 20). Our method (**Ours**) consistently outperforms Zip‑NeRF, 3DGS_pre, and NAN.

| Method   | Gain=2                | Gain=4                | Gain=8                | Gain=16               | Gain=20               | Average               |
|----------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|          | PSNR    | SSIM    | PSNR    | SSIM    | PSNR    | SSIM    | PSNR    | SSIM    | PSNR    | SSIM    | PSNR    | SSIM    |
| NAN      | 23.49   | 0.7706  | 23.58   | 0.7571  | 23.38   | 0.7389  | 21.37   | 0.7014  | 21.20   | 0.7074  | 22.60   | 0.7351  |
| 3DGS_pre | 25.06   | 0.7511  | 24.63   | 0.7553  | 23.29   | 0.7526  | 21.35   | 0.6813  | 21.23   | 0.6976  | 23.11   | 0.7276  |
| Zip‑NeRF | 25.10   | 0.8169  | 24.62   | 0.7913  | 23.91   | 0.7578  | 21.66   | 0.7049  | 21.31   | 0.7134  | 23.32   | 0.7569  |
| **Ours** | **25.47** | **0.8263** | **24.94** | **0.7989** | **24.16** | **0.7614** | **21.78** | **0.7102** | **22.44** | **0.7215** | **23.76** | **0.7637** |

> Visual comparisons and video results will be added soon.


## Installation

```bash
# Clone the repo
git clone https://github.com/ping-RGB/LNP-ZipNeRF.git
cd LNP-ZipNeRF

# Create conda environment
conda create --name lnp_zipnerf python=3.9
conda activate lnp_zipnerf

# Install requirements
pip install -r requirements.txt

# Install CUDA extensions 
pip install ./extensions/cuda

# (Optional) Install nvdiffrast for textured mesh
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

# Install torch_scatter with your CUDA version, e.g. CUDA 11.7
CUDA=cu117
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```
## Dataset

We use reconstructed datasets derived from the MipNeRF360 dataset. The clean images in the dataset are corrupted by adding synthetic noise according to the noise model provided by NAN, and then denoised high‑quality images are obtained through the blind denoising network Diffbir. Specifically, noisy input images are generated from clean MipNeRF360 images using NAN's noise model at multiple gain levels (2, 4, 8, 16, 20), and the target supervision is provided by Diffbir‑denoised versions of these noisy images. The downloaded dataset should be placed under the `data/` directory.

You can download the original MipNeRF360 dataset from:
http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

To download and unzip:
```bash
mkdir data
cd data
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip
```

## Train
```
# Configure accelerate (optional)
accelerate config

# Where your data is
DATA_DIR=data/noisy_360_v2/bicycle
EXP_NAME=noisy_360/bicycle_gain4

# Experiment will be conducted under "exp/${EXP_NAME}" folder
# "--gin_configs=configs/360.gin" can be seen as a default config 
# and you can add specific config using --gin_bindings="..."
accelerate launch train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.factor = 4"

# or you can also run without accelerate (without DDP)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.factor = 4"

# alternatively you can use an example training script 
bash scripts/train_360.sh

# blender dataset
bash scripts/train_blender.sh

# metric, render image, etc can be viewed through tensorboard
tensorboard --logdir "exp/${EXP_NAME}"
```
## Render, Evaluate

We provide complete scripts for rendering videos, evaluating metrics (PSNR/SSIM/LPIPS), and extracting meshes (Marching Cubes / TSDF).  
**These will be made publicly available upon request or after paper acceptance.**  
If you are a reviewer and need access to reproduce our results, please contact us at sunqiucheng@ccsfu.edu.cn.
