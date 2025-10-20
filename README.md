🩺 AI-Based Diabetes Diagnosis & Retinopathy Detection System

📋 Description
This project is an AI-powered healthcare assistant designed to predict and diagnose diabetes and its complications using Deep Learning (CNN) and Machine Learning (Random Forest, XGBoost) models.
It performs three main diagnostic tasks:

1. Diabetes Prediction – Detects diabetes risk using patient medical data (e.g., glucose, BMI, insulin, etc.).
2. Gestational Diabetes Prediction – Predicts gestational diabetes risk for pregnant women.
3. Diabetic Retinopathy Detection – Analyzes retina images to detect diabetic retinopathy and classify disease severity.

All models are integrated into a single Flask web application, allowing users to either upload medical data or retina images to receive instant diagnostic predictions.

---

🔬 Model Overview

👁️ Diabetic Retinopathy Detection

* Two-stage hybrid architecture combining CNN (for feature extraction) and Random Forest (for classification).
* Stage 1: Binary model to detect presence/absence of diabetic retinopathy.
* Stage 2: Multi-class model to classify severity as Mild, Moderate, Severe, or Proliferative.
* Achieved ~70% accuracy overall.

💉 Diabetes Prediction

* Implemented XGBoost classifier trained on medical features.
* Achieved 97% accuracy in predicting diabetes risk.

🤰 Gestational Diabetes Prediction

* Built another XGBoost model specialized for pregnant women.
* Achieved 98% accuracy in identifying gestational diabetes risk.

---

⚙️ Tech Stack
Python 3.9+
TensorFlow / Keras
Scikit-learn
XGBoost
Flask
OpenCV
NumPy, Pandas, Matplotlib

---

🚀 Features

* AI-powered multi-model diagnosis system
* Supports both image and numerical data inputs
* Deep Learning + Machine Learning hybrid modeling
* Flask-based interactive web interface
* Instant diagnostic predictions for multiple diabetes-related conditions

---

🧩 Project Structure

app.py                     → Flask web app integrating all models
models/                    → Contains saved ML & DL models (.h5, .pkl)
static/                    → CSS, JS, and uploaded images
templates/                 → HTML templates for web UI
dataset/                   → Image and tabular datasets used for training
notebooks/                 → Jupyter notebooks for model training
requirements.txt           → Project dependencies

---

⚡ How to Run the Project

1. Clone the repository
   git clone [https://github.com/YourUsername/Diabetes-Diagnosis-System.git](https://github.com/YourUsername/Diabetes-Diagnosis-System.git)
   cd Diabetes-Diagnosis-System

2. Install dependencies
   pip install -r requirements.txt

3. Run the Flask app
   python app.py

4. Open in your browser
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

💡 Future Enhancements

* Integrate with hospital EMR systems for real patient testing
* Add a dashboard for tracking patient history and progress
* Deploy as a cloud-based API or mobile app
* Enhance feature extraction using pre-trained CNNs (e.g., VGG16, ResNet)
* Add a chatbot assistant for interactive diagnosis

---

🧑‍💻 Author
Norahan Alla
AI & ML Developer | Passionate about Healthcare AI & Intelligent Diagnostic Systems



