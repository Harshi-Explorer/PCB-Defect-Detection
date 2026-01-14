# PCB Defect Detection

PCB Defect Detection is a computer vision–based project that identifies defects in Printed Circuit Boards (PCBs) using **image processing techniques**, **CNN-based deep learning models**, and a **Streamlit web application** for real-time inference.

This project helps automate PCB inspection, reduce human error, and improve quality control in electronics manufacturing.

---

## Problem Statement

Manual inspection of Printed Circuit Boards is time-consuming, error-prone, and inefficient for large-scale production.  
The objective of this project is to **automatically detect and classify PCB defects** using image-based analysis and deep learning techniques.

---

## Objectives

- Preprocess PCB images using image processing techniques
- Train CNN-based deep learning models for defect classification
- Compare multiple CNN architectures
- Evaluate model performance using accuracy, confusion matrix, and classification report
- Deploy the best-performing model using a Streamlit web interface

---

## Technologies Used

- **Programming Language:** Python  
- **Image Processing:** OpenCV, NumPy  
- **Deep Learning:** TensorFlow / PyTorch, CNN models  
- **Visualization:** Matplotlib, Seaborn  
- **Web Application:** Streamlit  
- **Platform:** Google Colab, GitHub  

---

## Dataset

The project uses a PCB defect dataset containing images of defective and non-defective PCBs.

### Defect Types (example)
- Missing hole  
- Mouse bite  
- Open circuit  
- Short circuit  
- Spur  
- Spurious copper  

*(Dataset is organized into train and test folders for model training and evaluation.)*

---


---

## Image Processing Techniques Used

- Image resizing
- Grayscale conversion
- Noise removal
- Normalization
- Data augmentation

These steps help improve model accuracy and robustness.

---

## Deep Learning Models Used

- Custom CNN
- Pretrained CNN models (ResNet / EfficientNet / others if used)

### CNN Advantages
- Automatic feature extraction
- High accuracy for image-based tasks
- Efficient learning for visual patterns

---

## Model Evaluation

The models are evaluated using:
- Accuracy
- Loss
- Confusion Matrix
- Classification Report

Graphs for training and validation accuracy/loss are generated for analysis.

---

## Streamlit Web Application

A Streamlit-based UI is developed to:
- Upload PCB images
- Perform real-time defect prediction
- Display predicted defect class

### Run Streamlit App

```bash
cd PCB_Dataset/streamlit_inference
pip install -r requirements.txt
streamlit run app.py
Results

CNN models successfully classify PCB defects

Streamlit application provides an easy-to-use interface

The system reduces manual inspection effort

Screenshots of results and UI are available in the screenshots/ folder.

Conclusion

This project demonstrates how image processing and deep learning can be effectively applied to automated PCB defect detection.
The Streamlit deployment makes the solution user-friendly and practical for real-world usage.

Future Scope

Integration with real-time camera systems

Use of object detection models (YOLO)

Deployment on cloud platforms

Expansion to industrial-scale datasets

Author

Harshitha Kothuru
B.Tech – Artificial Intelligence & Machine Learning


