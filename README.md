# VSRGAN
*A GAN-based approach for 360p → 2K Video Enhancement*

This repository contains an implementation and analysis of **SRGAN** (Super-Resolution Generative Adversarial Network) adapted for **video super-resolution (VSR)**. The project enhances low-resolution videos using perceptual loss and adversarial training to recover rich, realistic textures.

---

## Overview

Traditional super-resolution methods optimized using **MSE** or **PSNR** metrics often produce smooth, blurry outputs.  
**SRGAN** overcomes this by combining:

- **Adversarial Loss** – encourages natural, realistic textures  
- **VGG-Based Perceptual Loss** – preserves high-level semantic detail  
- **Residual Blocks + PixelShuffle** – efficient ×4 spatial upscaling  

We apply SRGAN frame-by-frame on video sequences, achieving much better perceptual quality than Bicubic or Lanczos interpolation.

---

## Methodology

### SRGAN Architecture

#### Generator
- Initial 9×9 convolution layer  
- **16 Residual Blocks** (Conv + BatchNorm + PReLU)  
- Two **PixelShuffle** (sub-pixel) layers for 4× upscaling  
- Final 9×9 convolution → high-resolution output  

#### Discriminator
- 8 convolutional layers (64 → 512 filters)  
- Fully connected layers  
- Sigmoid output for real/fake classification  

---

##  Perceptual Loss

The total perceptual loss is combination of:

- **Content Loss**: Euclidean distance between VGG-19 feature maps of HR and SR images  
- **Adversarial Loss**: pushes generator toward producing photorealistic textures  

--- 





