---

# 🍫 Cocoa Detection & Classification Models

This project covers two primary tasks: detecting cocoa readiness in images using **YOLOv8** and classifying those images with **EfficientNetB0**. Both models aim to aid in cocoa farming by providing actionable insights into the maturity of cocoa crops.

---

## 📚 Overview

### 🍃 Task 1: Cocoa Detection with YOLOv8
We use the **YOLOv8 object detection model** to predict cocoa readiness by detecting objects in images. This notebook demonstrates how to train and fine-tune the model for optimal performance.

### 🧠 Task 2: Cocoa Classification with EfficientNetB0
To classify the detected cocoa pods into different stages of readiness, we fine-tuned **EfficientNetB0**, a lightweight yet powerful convolutional neural network, optimized for mobile deployment.

---

## ⚙️ Workflow

### 🍫 YOLOv8 Cocoa Detection

1. **Environment Setup**: Install the required dependencies and ensure access to a GPU for training.
2. **Data Download & Preparation**:
   - **Dataset**: Cocoa image dataset (`cocoa_new.zip`).
   - **Labels**: Convert label annotations into YOLO format for training.
3. **Model Training**:
   - Use the YOLOv8 architecture for object detection.
   - Choose the appropriate model size (`yolov8s.pt`, `yolov8x.pt`, etc.).
4. **Evaluation**: Measure the performance of the trained model and visualize results.
5. **Fine-Tuning (Optional)**: Adjust hyperparameters such as learning rate and batch size to enhance model performance.

### 🌿 EfficientNetB0 Cocoa Classification

1. **Exploratory Data Analysis (EDA)**: Analyze the distribution and balance of the dataset. Visualize sample images to understand the data better.
2. **Model Selection**:
   - **EfficientNetB0** was selected due to its high accuracy and small size, making it ideal for mobile deployment.
   - Other architectures like **ResNet50** and **VGG19** were explored but rejected due to their larger size and slower inference times.
3. **Training**:
   - **Feature Extraction**: Train the model on custom cocoa dataset.
   - **Fine-Tuning**: Unfreeze the top layers of the model and train them for higher accuracy on the cocoa dataset.
4. **Callbacks**: Utilize key callbacks like:
   - **Mixed Precision Training**: For faster computation on supported GPUs.
   - **Early Stopping & ReduceLROnPlateau**: To prevent overfitting and help the model converge efficiently.

---

## 🔧 Model Parameters

- **YOLOv8 Cocoa Detection**:
  - Epochs: 100+
  - Batch size: 16
  - Input size: 640x640 pixels
- **EfficientNetB0 Cocoa Classification**:
  - Image size: 448x448
  - Batch size: 32
  - Layers fine-tuned: 118
  - Dropout: 0.2
  - Callbacks: Early stopping, ReduceLROnPlateau, TensorBoard

---

## 🛠️ How to Run

### 🍫 YOLOv8 Cocoa Detection
1. **Install Dependencies**: Ensure that required packages are installed by running the setup cell in the notebook.
2. **Download Dataset**: Execute the provided code to download and extract the `cocoa_new.zip` dataset.
3. **Generate Labels**: Convert the dataset annotations into YOLO format.
4. **Train Model**: Choose your desired YOLOv8 model and train using the provided cells.
5. **Evaluate & Fine-Tune**: Adjust hyperparameters for better accuracy, and run evaluation cells to measure performance.

### 🌿 EfficientNetB0 Cocoa Classification
1. **Preprocessing**: Ensure that the data is balanced and correctly formatted.
2. **Training**: Run the training cell for EfficientNetB0 using mixed precision training and early stopping to get the best results.
3. **Fine-Tuning**: Fine-tune the top layers of the model for improved accuracy.
4. **Submission**: Prepare predictions and generate the final CSV for submission.

---

## 📊 Results

- **YOLOv8 Detection Performance**: Achieved high accuracy in detecting cocoa pods and readiness stages.
- **EfficientNetB0 Classification Performance**: Performed better than ResNet50 and VGG19, with the added benefit of a smaller model size, making it ideal for mobile deployment.

---

## 📝 Next Steps

1. **Address Class Imbalance**: Explore methods such as **SMOTE** or **upsampling** to resolve imbalances in the training data.
2. **Optimize Data Pipeline**: Implement an efficient data loading and augmentation pipeline to speed up training and address memory issues.
3. **Model Deployment**: Deploy the EfficientNetB0 model in a mobile app for real-time cocoa classification on the field.

---

## 🤝 Contributions

Feel free to contribute by opening issues or submitting pull requests. Collaboration is encouraged to improve both the detection and classification pipelines.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

🌿 **Fine-tune your models, and let's help farmers predict the readiness of cocoa crops with the power of AI!**

---
