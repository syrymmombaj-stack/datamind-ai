# DataMind AI

**[Live Streamlit demo](https://datamind-ai-mpqbquetvscgmywhi72kapp.streamlit.app/)** · [Source code](https://github.com/syrymmombaj-stack/datamind-ai)

An interactive, end-to-end tabular machine learning demo. Upload a CSV or explore three bundled datasets, inspect missing values, choose a target, compare a baseline with linear and random-forest models, inspect permutation feature importance, and download predictions for new rows.

## Included datasets

| Dataset | Task | Rows | Target | Source |
| --- | --- | ---: | --- | --- |
| Wine varieties | Classification | 178 | `wine_class` | [scikit-learn / UCI Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| Iris flowers | Classification | 150 | `species` | [scikit-learn Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) |
| Diabetes progression | Regression | 442 | `disease_progression` | [scikit-learn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) |

The CSV files are in `data/` and can be downloaded from the app. Diabetes features are rounded to six decimal places in the bundled CSV. These small teaching datasets demonstrate the workflow; diabetes predictions are for education, not medical use.

**No paid APIs, credits, or keys required.** Built with Python, pandas, scikit-learn, and Streamlit.

**Portfolio case study:** [Wine classification — question, method, metrics, and limitations](CASE_STUDY.md).

## Custom website

The new independent website is in `web/` and runs with `server.py`. It uses the same `src/ml.py` pipeline and the same bundled datasets as the Streamlit demo. Python serves the responsive HTML/CSS/JavaScript interface and a same-origin JSON API for dataset inspection, training, and prediction. No external model API or API key is required.

```bash
pip install -r requirements-web.txt
python server.py
# open http://localhost:8000
```

`render.yaml` configures a free Render web service. Connect this public repository to Render as a Blueprint; the build command installs `requirements-web.txt` and the start command runs `python server.py`. The free instance sleeps when idle, so the first request after inactivity can take about a minute. A custom paid domain is optional; Render supplies a free `onrender.com` address.

For this public demo, CSV files are limited to 10 MB, 3,000 rows, and 40 columns. Prediction files are limited to 1 MB and 1,000 rows. Trained models stay in memory for up to 30 minutes or until the service restarts; no persistent account or storage is provided. Do not upload private or sensitive datasets.

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
