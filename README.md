# DataMind AI

**[Live demo](https://datamind-ai-mpqbquetvscgmywhi72kapp.streamlit.app/)** · [Source code](https://github.com/syrymmombaj-stack/datamind-ai)

An interactive, end-to-end tabular machine learning demo. Upload a CSV or explore three bundled datasets, inspect missing values, choose a target, compare a baseline with linear and random-forest models, inspect permutation feature importance, and download predictions for new rows.

## Included datasets

| Dataset | Task | Rows | Target | Source |
| --- | --- | ---: | --- | --- |
| Wine varieties | Classification | 178 | `wine_class` | [scikit-learn / UCI Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| Iris flowers | Classification | 150 | `species` | [scikit-learn Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) |
| Diabetes progression | Regression | 442 | `disease_progression` | [scikit-learn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) |

The CSV files are in `data/` and can be downloaded from the app. Diabetes features are rounded to six decimal places in the bundled CSV. These small teaching datasets demonstrate the workflow; diabetes predictions are for education, not medical use.

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
- All models use the same 75/25 split and random seed. Classification splits are stratified. Three-fold cross validation on the training partition chooses the winner (macro F1 for classification, MAE for regression). An untouched test partition is used once for final reporting. Preprocessing happens inside each cross validation fold.
- Permutation importance is computed on up to 300 holdout rows. This measures score change after shuffling a column; it does not prove causation.

## Limits and responsible use

This is a portfolio prototype, not a validated decision system. Three-fold cross validation has high variance on small datasets; a real deployment needs more extensive validation, drift monitoring, and domain review. Small, imbalanced, high-cardinality, time series, sensitive, or personally identifying datasets require additional care. Never upload private data to a public hosted demo. Predictions are produced in memory and are not stored by this app.

## Project layout

`app.py` is the UI. `src/ml.py` contains reusable preprocessing, evaluation, importance, and prediction functions. `tests/test_ml.py` checks training and unseen-category predictions.
