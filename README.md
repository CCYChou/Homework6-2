# HW6-2: Face Mask Detection Using Transfer Learning

This project demonstrates a transfer learning approach using VGG16 and MobileNetV2 models for binary classification of images into "with_mask" and "without_mask" classes. The CRISP-DM methodology is followed to ensure a structured implementation.

---

## 1. Business Understanding

### Objective:
To classify images as `with_mask` or `without_mask` using pre-trained deep learning models. This solution aims to aid in public safety measures by identifying individuals wearing or not wearing a mask.

### Applications:
- Public health compliance checks.
- Monitoring in restricted areas or crowded public spaces.

---

## 2. Data Understanding

### Dataset:
- The dataset is sourced from the GitHub repository: [Face Mask Detection](https://github.com/chauhanarpit09/Face-Mask-Detection-).
- Data is divided into:
  - `Train`: Training images.
  - `Validation`: Validation images.
  - `Test`: Testing images.

### Characteristics:
- Images are categorized into two binary classes: `with_mask` and `without_mask`.
- Synthetic augmentation is used to expand dataset diversity.

---

## 3. Data Preparation

### Steps:
1. **Dataset Cloning**:
   - Clone the dataset repository:
     ```bash
     git clone https://github.com/chauhanarpit09/Face-Mask-Detection-.git
     ```

2. **Directory Verification**:
   - The script verifies dataset structure to ensure proper paths for training, validation, and testing.

3. **Data Augmentation**:
   - Augmentation applied includes rotation, shifting, zooming, and flipping using `ImageDataGenerator` to improve model generalization.

4. **Rescaling**:
   - Pixel values are normalized to the range `[0, 1]` for consistency.

---

## 4. Modeling

### VGG16 Model:
1. **Base Model**:
   - Pre-trained VGG16 with `imagenet` weights.
   - Top layers excluded (`include_top=False`).
   - Base layers frozen to preserve pre-trained knowledge.

2. **Custom Layers**:
   - Fully connected layers with ReLU activation.
   - Dropout for regularization.
   - Sigmoid activation for binary classification.

3. **Compilation**:
   - Optimizer: Adam.
   - Loss: Binary Cross-Entropy.
   - Metric: Accuracy.

### MobileNetV2 Model:
1. **Base Model**:
   - MobileNetV2 pre-trained on `imagenet`.

2. **Custom Layers**:
   - Global Average Pooling.
   - Dense layers with ReLU and Sigmoid activations.

3. **Compilation**:
   - Optimizer: Adam.
   - Loss: Binary Cross-Entropy.
   - Metric: Accuracy.

---

## 5. Evaluation

### Metrics:
- Training and validation accuracy.
- Training and validation loss.

### Testing:
- The model is evaluated on the test dataset to compute accuracy and loss.
- Example input images are tested via URL inputs for prediction.

### Results:
- Metrics are displayed for both training and testing phases.

---

## 6. Deployment

### Usage:
1. **Run the Script**:
   - For Python script (`hw6_2.py`):
     ```bash
     python hw6_2.py
     ```
   - For Notebook (`HW6_2.ipynb`):
     Open in Jupyter or Colab and execute the cells sequentially.

2. **Test with Image URL**:
   - Input an image URL to classify:
     ```python
     image_url = "https://example.com/image.jpg"
     test_image(image_url, model, ['without_mask', 'with_mask'])
     ```

---

## Requirements

### Libraries:
- TensorFlow
- NumPy
- Pillow
- Matplotlib

### Installation:
```bash
pip install tensorflow numpy pillow matplotlib
