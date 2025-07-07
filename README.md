# 🤖 Customer Query Flow Mapping & Chatbot Optimization

This project analyzes historical chatbot interactions to improve customer query handling and reduce fallback rates. It involves:

- Intent detection using ML
- Query flow visualization
- Fallback identification & resolution
- Chatbot logic optimization

## 📂 Structure

- `data/`: Chat logs (anonymized)
- `notebooks/`: EDA & modeling
- `scripts/`: Model training scripts
- `updated_model/`: Trained intent classifier (`.pkl`)
- `visuals/`: Visio/Draw.io flow diagrams
- `report/`: Summary & performance results

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python scripts/train_model.py
