# CCTV Abnormal Behavior Detection

## 📌 Project Overview
Detect abnormal behaviors (invasion, loitering, arson) using both rule-based logic and AI-based models.

## 🚀 Structure
- Rule-based logic handles clear cases
- AI model (e.g., MIL, 3D-CNN + LSTM) for ambiguous patterns
- Modular structure for easy testing, GUI integration, and model training

## 🔧 How to Run
```bash
python main.py --mode main
```

## 📁 Directory Overview
- `config/`: system config
- `models/`: AI models
- `yolo/`: person detection
- `preprocessing/`: logic-based rule evaluation
- `gui/`: real-time GUI interface
- `train/`: model training scripts
- `inference/`: testing pipeline
