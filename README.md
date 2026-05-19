# 🌾 SmartAgriML: AI-Powered Crop Failure Risk Prediction

SmartAgriML is a professional Machine Learning system designed to predict the risk of crop failure and yield decline based on environmental and agricultural parameters. Built with real-world scenarios in mind, this project demonstrates a complete end-to-end Machine Learning pipeline accessible via an interactive modern dashboard.

## 🌟 Key Features
- **Exploratory Data Pipeline**: Custom validation and handling of anomalies. 
- **Advanced Model Training**: Evaluates Random Forest, Gradient Boosting, and XGBoost using `RandomizedSearchCV` for aggressive hyperparameter tuning.
- **Explainable AI (XAI)**: Includes built-in feature importance rendering so farmers (and reviewers) know *exactly* what factors strictly drove the risk rating.
- **High-Performance Backend API**: Leverages FastAPI for extremely fast, concurrent prediction routing.
- **Modern Dashboard UI**: Built on Streamlit, featuring an in-memory session history, responsive KPI tracking layout, and Altair graphical analysis charts.

---

## 🏗️ Architecture Overview

The project is heavily decoupled into 4 major aspects:

1. **Configurations (`config.yaml`)**: Stores paths and search CV parameters.
2. **Pipelines (`src/pipeline/`)**: Handles logical orchestration. Includes `train_pipeline.py` and `predict_pipeline.py`.
3. **Core ML Components (`src/components/`)**:
   - `data_ingestion.py`: Connects to standard data sources.
   - `data_validation.py`: Ensures strict bounds logic and missing value imputation.
   - `data_transformation.py`: Scalers and categorical encodings (OneHot).
   - `model_trainer.py`: Tunes and saves the best model automatically.
4. **App & API (`app.py`, `main.py`)**: Seamless interaction layers serving predictions immediately upon request.

---

## 🚀 Execution Steps (How to Run)

### 1. Prerequisites
Ensure you have `Python 3.10+` installed.

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model Training (Optional, if you wish to retrain)
```bash
python -m src.pipeline.train_pipeline
```
*This will rebuild and save the tuned model artifacts dynamically referencing `config.yaml`.*

### 4. Running the Backend API
In a new terminal window:
```bash
uvicorn main:app --reload --port 8000
```
*(The API will be available at http://localhost:8000. Interactive Swagger UI is at http://localhost:8000/docs)*

### 5. Running the User Dashboard (Frontend)
In a secondary terminal window:
```bash
streamlit run app.py
```
*(The UI will launch on http://localhost:8501)*

---

## 🛠️ Technology Stack
- **Modelling**: Scikit-Learn, XGBoost, Pandas, NumPy
- **Serving**: FastAPI, Uvicorn, Pydantic
- **UI & Viz**: Streamlit, Altair
- **Architectural Ops**: Standard Python logging, exceptions handling, yaml parsing

---
*Created as part of a Placement Portfolio - For detailed interview preparation and project Q&A context, see [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md).*
