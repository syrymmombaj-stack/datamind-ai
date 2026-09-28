# DataMind AI

An interactive, end-to-end tabular machine learning demo. Upload a CSV or explore the built-in iris sample, inspect missing values, choose a target, compare a baseline with linear and random-forest models, inspect permutation feature importance, and download predictions for new rows.

**No paid APIs, credits, or keys required.** Built with Python, pandas, scikit-learn, and Streamlit.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free

Push this folder to a GitHub repository, sign in to Streamlit Community Cloud, select the repository and `app.py`, then deploy. Free hosting may have resource and inactivity limits. No payment method or external model API is needed for this implementation.

## Method

- Rows with missing targets are removed; numeric missing features are imputed with medians, categorical features are one-hot encoded. Preprocessing is fitted inside each training pipeline, after the train/test split.
- Classification: most-frequent baseline, logistic regression, random forest; select highest macro F1. Regression: mean baseline, ridge, random forest; select lowest MAE.
- All models use the same 75/25 holdout and random seed. Classification splits are stratified. Accuracy or R² and other applicable metrics appear alongside the selection metric.
- Permutation importance is computed on up to 300 holdout rows. This measures score change after shuffling a column; it does not prove causation.

## Limits and responsible use

This is a portfolio prototype, not a validated decision system. The displayed holdout is used to select a model, so its winning score may be optimistic. For a real deployment, use cross validation for selection and an untouched final test set. Small, imbalanced, high-cardinality, time series, sensitive, or personally identifying datasets require additional care. Never upload private data to a public hosted demo. Predictions are produced in memory and are not stored by this app.

## Project layout

`app.py` is the UI. `src/ml.py` contains reusable preprocessing, evaluation, importance, and prediction functions. `tests/test_ml.py` checks training and unseen-category predictions.
