# 🧠 DeepFake Identifier (AI vs Real Image Classifier)

This project is a **Deep Learning-based Image Classifier** that detects whether an image is:

- 🖼️ Real Image  
- 🤖 AI Generated Image  

The model is built using **PyTorch** and deployed using **Streamlit**.

---

## 🚀 Features

- Upload JPG, JPEG, PNG, JFIF images
- Predict whether image is AI-generated or real
- Displays prediction with confidence score
- Simple and clean Streamlit UI

---

## 🏗️ Model Details

- Custom CNN Architecture
- 3 Convolutional Layers
- MaxPooling Layers
- Fully Connected Layers
- Trained on AI vs Real image dataset
- Framework: PyTorch
---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/santhosh18-M/DeepFake-Identifier.git
cd DeepFake-Identifier

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py