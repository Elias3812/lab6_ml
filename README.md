# Project Overview

This repository hosts the code for the paper **["Boundary-Aware Segmentation Network for Mobile and Web Applications"](https://arxiv.org/pdf/2101.04704)** by Qin et al., published in 2019. The paper introduces a novel deep learning-based framework for Salient Object Detection (SOD), emphasizing boundary awareness to enhance segmentation precision. The approach outperforms prior methods by leveraging a hybrid loss function and a carefully designed network architecture that balances global and boundary information.

## Paper Reference

Qin, X., Zhang, Z., Huang, C., Dehghan, M., & Zaiane, O. (2019). **BASNet: Boundary-Aware Salient Object Detection**. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

## Topics Covered

- Salient Object Detection (SOD)
- Boundary Awareness in Image Segmentation
- Deep Learning Architectures
- Hybrid Loss Functions

---

## Key Contributions

### Research Goals and Results

The BASNet framework addresses the problem of accurately detecting salient objects in images, particularly focusing on refining the boundaries of the detected regions. The proposed network architecture combines global feature extraction with boundary refinement to deliver state-of-the-art performance.

### Algorithm/Idea
The model introduces a **Boundary-Aware Neural Network (BASNet)**, which consists of a densely supervised encoder-decoder architecture. It incorporates a hybrid loss function composed of:
- Binary Cross-Entropy
- Structural Similarity
- A new IoU-based loss term

### Advantages
- The integration of boundary information ensures precise segmentation, particularly at object edges.
- BASNet outperforms existing methods on benchmark datasets like DUTS, ECSSD, and HKU-IS, achieving better **F-measure** and lower **mean absolute error**.

### Novelty
The boundary-aware loss function and hybrid supervision strategy mark a significant improvement over traditional SOD models that often struggle with edge accuracy.

---

## Running the Code

### Requirements

Before starting the training or testing process, ensure you are working within the correct Python environment. Follow these steps to activate the `venv` with Python 3.6:

1. Navigate to your project directory in the terminal.
2. Create and activate a Python 3.6 virtual environment:

```bash
# Create a virtual environment
py -3.6 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install the dependencies using:
```bash
pip install -r requirements.txt
```
## Dataset Preparation

Download the benchmark dataset used for training and evaluation:

- **DUTS Dataset**: [Download Link](https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset?resource=download-directory)

Organize the dataset into the following structure:

```plaintext
train_data/
  DUTS-TR/
    DUTS-TR-Image/
    DUTS-TR-Mask/

test_data/
  test_images/
  test_results/
    test_images/
```

## Training the Model

1. Modify the configuration file to set dataset paths and hyperparameters.
2. Train the model using the following command:

```bash
python train.py
```
Model will be saved into the directory 'saved_models/basnet_bsi/'.

Also you can ownload the pre-trained model basnet.pth from [GoogleDrive](https://drive.google.com/file/d/1K9y9HpupXT0RJ4U4OizJ_Uk5byUyCupK/view)  extraction code, and put it into the directory 'saved_models/basnet_bsi/'

## Testing the Model
To test the pre-trained model:

1. Place your test images in the test_data/test_images/ directory.
2. Run the test script:
```bash
python test.py
```
The results will be saved in the test_data/test_results/test_images/ directory.


## Exploring the Code
### Key Components
- Model Architecture: The encoder-decoder network with skip connections ensures effective multi-scale feature fusion and boundary refinement.
- Loss Function: The hybrid loss combines global and local constraints, specifically designed to enhance boundary precision.
- Training Pipeline: Includes data augmentation, optimizer setup, and model checkpointing for efficient training.

## Suggestions for Improvement
- Preprocessing: Implement advanced data augmentation techniques such as CutMix or MixUp to improve generalization.
- Data Sources: Utilize additional datasets for diverse training samples.
- Model Architecture: Explore lightweight versions of the model for real-time applications by introducing depthwise separable convolutions.
- Post-Processing: Use CRF (Conditional Random Fields) to refine boundaries further.
