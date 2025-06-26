# 💍 Dowry Demand Prediction

Predict dowry demand categories in Bangladesh marriages using machine learning and modern APIs.

---

## 🚀 Overview

Dowry is a critical social issue in South Asia, especially in Bangladesh. This project leverages data science and machine learning to predict dowry demand categories based on socio-economic and demographic features. The solution includes:

- **Data cleaning & preprocessing notebooks**
- **Feature engineering & model training**
- **API deployment with FastAPI**
- **Interactive web app with Streamlit**
- **Ready for cloud deployment (Render, etc.)**

---

## 📊 Features

- **End-to-end ML pipeline:** From raw Excel data to deployed API.
- **Robust data cleaning:** Outlier removal, skewness correction, and normalization.
- **Feature engineering:** One-hot encoding, log transforms, and more.
- **Model selection:** Random Forest, XGBoost, Logistic Regression, KNN.
- **Class imbalance handling:** Oversampling with `imblearn`.
- **API:** FastAPI endpoint for real-time predictions.
- **Web App:** Streamlit UI for easy user interaction.
- **Cloud-ready:** Easy deployment on Render or similar platforms.

---

## 🗂️ Project Structure

```
Dowry-Demand-Prediction/
│
├── Notebooks/
│   ├── data_clean.ipynb            # Data cleaning & outlier handling
│   ├── data_preprocessing.ipynb    # Feature engineering & encoding
│   ├── model_training.ipynb        # Model training & evaluation
│   └── data_visualization.ipynb    # Data visualization (optional)
│
├── Dataset/
│   ├── raw/                        # Raw Excel files
│   ├── cleaned/                    # Cleaned data
│   └── encoded/                    # Encoded data for ML
│
├── models/
│   ├── random_forest_model.pkl     # Trained ML model
│   └── dowry_label_encoder.pkl     # Label encoder for target
│
├── api/
│   └── main.py                     # FastAPI app
│
├── streamlit_app/
│   └── app.py                      # Streamlit web app
│
├── requirements.txt
├── main.py                         # Entrypoint for API
├── README.md
└── LICENSE
```

---

## 🧑‍💻 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Lokesh-DataScience/Dowry-Demand-Prediction.git
cd Dowry-Demand-Prediction
```

### 2. Set up the environment

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# Or
source .venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt
```

### 3. Run the API

```bash
python main.py
# or
uvicorn api.main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive API docs.

### 4. Run the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

---

## 🧠 Model Details

- **Input features**: Year, ages, family type, area, jobs, marriage type, marital status, mohor amount, etc.
- **Preprocessing**: Outlier removal, log transforms, one-hot encoding.
- **Class imbalance**: Handled with RandomOverSampler from `imblearn`.
- **Best models**: Random Forest and XGBoost (highest accuracy and F1-score).

---

## 🌐 API Usage

### POST `/predict`

#### Request Example:

```json
{
  "year": 2013,
  "womens_age": 18.0,
  "mens_age": 25.0,
  "mohor_amount": 420000,
  "family_type": "Higher Class",
  "area": "Satkhira",
  "girls_job": "Day Labour",
  "boys_job": "Business/Entrepreneur",
  "marriage_type": "Forced Marriage",
  "womens_marital_status": "Single"
}
```

#### Response Example:

```json
{
  "predicted_dowry": "Aurnaments",
  "confidence": 0.4,
  "timestamp": "2025-06-26T18:53:02.063803",
  "input_summary": {
    "year": 2013,
    "womens_age": 18.0,
    "mens_age": 25.0,
    "age_difference": 7.0,
    "mohor_amount": 420000.0,
    "family_type": "Higher Class",
    "area": "Satkhira",
    "marriage_type": "Forced Marriage"
  }
}
```

---

## 📈 Results

- **Random Forest Accuracy**: ~98%
- **XGBoost Accuracy**: ~96%
- **Handles class imbalance and multi-class prediction robustly.**

---

## ☁️ Deployment

- **Render**: Ready for one-click deployment.  
  See [Render Python Deploy Docs](https://render.com/docs/deploy-python).
- **Docker**: Easily containerizable for any cloud.

---

## 🙌 Contributing

Pull requests, issues, and suggestions are welcome!  
Please open an issue or submit a PR.

---

## 📄 License

MIT License

---

## ⭐ Acknowledgements

- Bangladesh dowry data sources  
- [scikit-learn](https://scikit-learn.org/)  
- [imbalanced-learn](https://imbalanced-learn.org/)  
- [XGBoost](https://xgboost.readthedocs.io/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
- [Streamlit](https://streamlit.io/)

---

## 🔗 Links

- [Project on GitHub](https://github.com/)  
- [FastAPI Docs](https://fastapi.tiangolo.com/)  
- [Streamlit Docs](https://docs.streamlit.io/)
