# HDA-Net: H&E and RGB Dual Attention Network for Nuclei Instance Segmentation (IEEE ACCESS 2024) [[PAPER](https://ieeexplore.ieee.org/document/10504830)]

The reproduction code of HDA-Net which is accepted in IEEE ACCESS 2024

## Abstract

H&E-stained images (HSIs) are widely adopted for revealing cellular structures and capturing morphological changes in nuclear instance segmentation. Despite several studies aimed at differentiating pixels of overlapping nuclei, challenges persist due to color inconsistencies introduced by non-uniform manual staining operations. This issue leads to blurred borders and alterations in color representation within the images. This study proposes an H&E and RGB dual attention network (HDA-Net) designed to address these challenges and enhance nuclei instance segmentation accuracy. Specifically, our approach involves decomposing hematoxylin, eosin, and residual (HER) components from HSIs using color deconvolution to extract discriminative information, such as nuclei and cytoplasm, entangled in RGB images. Additionally, we propose an H&E and RGB dual attention module (HDA) and a single-source attention module (SAM) to effectively incorporate information from the decomposed HER to enhance RGB images. The HDA module utilizes cross-attention between RGB features and decomposed HER to enhance RGB representation under the guidance of the HER components. Conversely, SAM applies global channel attention to learn the correlations between channels of each RGB and HER component. In the final decoder stage, the study employs multi-task learning to capture region and centroid information of the label, providing comprehensive supervision for clustered and overlapped nuclei. Our experiments on the CoNSeP, PanNuke, and Kumar datasets demonstrated that the proposed HDA-Net outperformed existing models, showing improvements of 1.2% in AJI and 0.76% in Dice for CoNSeP, 0.7% and 0.5% for PanNuke, and 0.21% and 0.16% for Kumar, respectively.

## Overall Architecture of HDA-Net
![HDANet](https://github.com/user-attachments/assets/24690788-e15f-49a8-bb66-d73eae778091)

## Experiment Results

### Seen Clinical Settings Results
![Screenshot from 2023-11-26 16-16-13](https://github.com/BlindReview922/MADGNet/assets/142275582/30767364-13a7-43b1-8b00-dff7aa531e7d)

### Unseen Clinical Settings Results
![Screenshot from 2023-11-26 16-15-44](https://github.com/BlindReview922/MADGNet/assets/142275582/cef29e7d-5c41-4c82-9f9a-45c45de46cb9)

# Bibtex

```
@article{im2024hda,
  title={HDA-Net: H\&E and RGB Dual Attention Network for Nuclei Instance Segmentation},
  author={Im, Yu-Han and Park, Seo-Hyeong and Lee, Sang-Chul},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
