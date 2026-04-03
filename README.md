# 🚨 Real-Time Threat Detection for Low-Light Intelligent Surveillance

An advanced Computer Vision and Deep Learning system designed to detect anomalies and potential threats in video surveillance footage, particularly optimized for low-light environments.

## 🌟 Key Features

* **Hybrid Model Architecture**: Combines **YOLOv8** for precise object detection and spatial frame analysis with a custom **LSTM** network for temporal sequence processing and identifying behavioral anomalies over time.
* **Low-Light Capability**: Engineered for robust detection even in poorly lit environments, ensuring effective 24/7 security monitoring.
* **Interactive Web Dashboard**: A clean, responsive interface built with Flask to easily upload surveillance videos and receive instant, intuitive threat analysis.
* **Performance Analytics**: Contains detailed metric evaluations (Accuracy, Precision, Recall, F1-Score) integrated directly into the web application to validate detection reliability.

## 🛠️ Technology Stack

* **Languages:** Python, JavaScript, HTML5, CSS3
* **AI / Deep Learning:** PyTorch, YOLOv8 (Ultralytics), OpenCV
* **Backend Framework:** Flask

## 📁 Project Structure

```text
├── app.py                     # Main Flask Application & Routing
├── config.py                  # Server Configuration & Parameters
├── model.py                   # PyTorch Model Architecture (LSTM)
├── evaluate.py                # Model Evaluation & Metrics generation
├── requirements.txt           # Python Project Dependencies
├── external_crime_lstm.pth    # Trained LSTM Model Weights
├── models/                    # Supplemental AI Model Artifacts
├── templates/                 # Frontend HTML Templates (Dashboard/Analysis)
└── uploads/                   # Video Ingestion Directory
```

## 🚀 Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/milimathew2005/-Real-Time-Threat-Detection-for-Low-Light-Intelligent-Surveillance-.git
   cd -Real-Time-Threat-Detection-for-Low-Light-Intelligent-Surveillance-
   ```

2. **Install the required dependencies:**
   Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the platform:**
   ```bash
   python app.py
   ```

4. **Access the Dashboard:**
   Open your web browser and navigate to `http://localhost:5000` to start analyzing surveillance videos!
