# Deep Dehazing Powered by Image Processing Network

This repository contains the implementation of the **Deep Dehazing Powered by Image Processing Network**, inspired by the core concepts of direct learning, curve adjustment, retinex decomposition, and multi-image fusion for effective single image dehazing. The overall model architecture, training process, and evaluation methods are based on the code provided above, with the structure optimized for ease of use and efficient training on RESIDE datasets.

## 📄 Description

This project aims to develop a single image dehazing model that integrates conventional image processing methods into a deep neural network, effectively combining the benefits of both approaches. It extracts features, applies direct dehazing, curve adjustment, and retinex decomposition, and fuses multiple intermediate results for accurate and robust dehazing.

## 📁 Dataset Information

### Data Loading and Preprocessing

The model uses the **RESIDE** dataset, which contains paired hazy and clear images:

* **Hazy Images:** `/kaggle/input/indoor-training-set-its-residestandard/hazy`
* **Clear Images:** `/kaggle/input/indoor-training-set-its-residestandard/clear`

Images are resized to 256x256 pixels and normalized for deep learning. The custom `RESIDEDataset` class in the code efficiently handles this preprocessing.

## 🏗️ Model Architecture

The model is composed of the following key components:

* **Feature Extractor:** Extracts initial features from hazy images using convolutional layers.
* **Direct Dehaze:** Directly estimates the dehazed image by adding learned features to the input.
* **Curve Adjustment:** Iteratively refines the dehazing results using curve maps.
* **Retinex Decomposition:** Separates illumination and reflectance components for enhanced dehazing.
* **Fusion Module:** Merges multiple intermediate outputs to generate the final dehazed image.

## 🔧 Hyperparameters

* **Batch Size:** 8
* **Learning Rate:** 1e-4
* **Num Epochs:** 50
* **Checkpoint Save Frequency:** 5 epochs
* **Loss Weights:** `λ1 = 0.5`, `λ2 = 1.0`, `λ3 = 0.5`

## 🚀 Training and Validation Process

The training process includes:

* **Data Loading:** Using the `RESIDEDataset` class
* **Loss Calculation:** Combining **L1 Loss**, **Perceptual Loss**, and **SSIM Loss**
* **Checkpoint Management:** Saving model states every 5 epochs
* **Visualization:** Periodic output previews for qualitative assessment

## 💡 Improvements and Future Updates

* Integrate more advanced architectures like ResNet and DenseNet
* Experiment with Transformer-based models for improved context understanding
* Add more robust data augmentation methods for enhanced generalization

## ⚠️ Notes and Cautions

### `pytorch_ssim` Library Modifications

1. **`padding` Type Issue Fix**

   * Ensure `padding` is an integer using:

   ```python
   padding = window_size // 2  # Use integer division
   ```
2. **Device Mismatch Handling**

   * Move `window` tensor to the correct device:

   ```python
   window = window.to(img1.device)
   ```

## 📝 Conclusion

This project effectively integrates traditional image processing techniques into a deep learning framework for high-quality single image dehazing. This approach highlights the importance of combining classic computer vision methods with modern deep learning techniques for enhanced performance.

## 🏆 Acknowledgments
This work was conducted with reference to the paper Deep Dehazing Powered by Image Processing Network (CVPRW 2023) by Guisik Kim, Jinhee Park, and Junseok Kwon. We would like to acknowledge their foundational contributions to this topic.
### [paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Kim_Deep_Dehazing_Powered_by_Image_Processing_Network_CVPRW_2023_paper.html)
