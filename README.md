# PCB Defect Detection

PCB defect detection using image processing, CNN-based deep learning models,
and a Streamlit web application.

## Project Structure
- PCB_Dataset/image_processing : Image preprocessing code and results
- PCB_Dataset/training : CNN model training scripts
- PCB_Dataset/inference : Standalone inference code
- PCB_Dataset/streamlit_inference : Streamlit app and inference logic

## Streamlit Usage
```bash
cd PCB_Dataset/streamlit_inference
pip install -r requirements.txt
streamlit run app.py
## Models Used
- Custom CNN
- Pretrained CNN architectures (e.g., ResNet, EfficientNet)
