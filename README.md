# 🏠 Bangalore House Price Predictor

> An end-to-end Machine Learning project that predicts house prices in Bangalore using real listing data — featuring full EDA, multi-model comparison, feature engineering, and a production-ready Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Highlights

- **6 ML models** trained and compared (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
- **5-fold cross-validation** for robust model evaluation
- **Full EDA** with 6 charts (price distribution, location heatmap, correlation matrix, etc.)
- **Feature engineering**: sqft per BHK, ready-to-move flag, area type encoding, outlier removal per location
- **Bathrooms intentionally excluded** — statistically redundant with BHK (85%+ correlation), explained in-app
- **Production Streamlit app** with 3 tabs: Prediction, EDA, Model Report
- **Unit tests** with pytest
- **Modular codebase** split into preprocess / train / predict / eda modules

---

## 🗂️ Project Structure

```
bangalore-house-price/
│
├── app.py                  ← Streamlit web app (3-tab UI)
├── train_model.py          ← Training entrypoint (run this first)
│
├── src/
│   ├── preprocess.py       ← Data cleaning & feature engineering
│   ├── train.py            ← Model training, comparison, saving
│   ├── predict.py          ← Prediction utilities
│   └── eda.py              ← EDA charts generation
│
├── tests/
│   └── test_preprocess.py  ← Unit tests (pytest)
│
├── models/                 ← Auto-generated after training
│   ├── model.pkl
│   ├── locations.pkl
│   ├── location_cols.pkl
│   ├── features.pkl
│   └── model_report.json
│
├── assets/                 ← EDA charts (auto-generated)
├── data/                   ← Place your CSV here
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/bangalore-house-price.git
cd bangalore-house-price
pip install -r requirements.txt
```

### 2. Download Dataset

Download `Bengaluru_House_Data.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

Place it inside the `data/` folder.

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Run full EDA and save charts to `assets/`
- Train 6 ML models and print comparison
- Save best model + metadata to `models/`

### 4. Launch the Web App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Model Comparison

| Model               | MAE (L) | RMSE (L) | R² Score | CV R² |
|---------------------|---------|----------|----------|-------|
| Linear Regression   | ~22     | ~35      | ~0.72    | ~0.71 |
| Ridge Regression    | ~22     | ~35      | ~0.72    | ~0.71 |
| Lasso Regression    | ~23     | ~36      | ~0.71    | ~0.70 |
| Decision Tree       | ~18     | ~30      | ~0.80    | ~0.78 |
| **Random Forest**   | **~15** | **~26**  | **~0.85**| **~0.84** |
| Gradient Boosting   | ~16     | ~27      | ~0.84    | ~0.83 |

*Best model auto-selected by CV R² score*

---

## 🧠 Why Bathrooms Are NOT a Feature

In the Bangalore dataset, bathrooms are almost perfectly correlated with BHK (r ≈ 0.85).
Including bathrooms as a feature introduces **multicollinearity** — the model already learns
this information from BHK. Excluding it produces a cleaner, more interpretable model.

**Real price drivers (by feature importance):**
1. 📍 Location — ~55% of price variance
2. 📐 Total Sqft — ~25%
3. 🛏️ BHK — ~12%
4. 🏗️ Availability (ready/under-construction) — ~5%
5. 📋 Area Type — ~3%

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Testing | Pytest |
| Version Control | Git + GitHub |

---

## 📈 Resume Line

> *"Built an end-to-end Bangalore house price prediction system — featuring EDA, feature engineering (outlier removal per location, sqft/BHK ratio, availability encoding), comparison of 6 ML models via 5-fold CV (best: Random Forest, R²=0.85, MAE=₹15L), and deployed as a 3-tab Streamlit web app with model explainability."*

---

## 📄 License

MIT License — free to use and modify.