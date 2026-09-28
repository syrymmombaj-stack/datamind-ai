# Case study: Classifying wine samples

**Live demo:** [DataMind AI](https://datamind-ai-mpqbquetvscgmywhi72kapp.streamlit.app/) — select **Wine varieties · classification**, then open **Train & evaluate**.

## Question

Can 13 measured chemical properties distinguish three categories of wine samples? This is a multiclass classification exercise using the 178-row [Wine dataset bundled with scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), derived from the UCI Wine dataset. The target is `wine_class`; the three categories contain 59, 71, and 48 samples. The dataset does not establish that predictions would generalize to new suppliers, years, instruments, or regions.

## Method

The app holds out 25% (45 rows) as a stratified test partition, with random seed 42. On the remaining 133 rows, it compares a most-frequent-class baseline, logistic regression, and a random forest using three-fold stratified cross validation. It selects the highest mean macro F1, fits that pipeline on the training partition, then reports test metrics. Missing-value imputation, numeric scaling, and categorical encoding live inside each pipeline and are fitted independently within each training fold.

Macro F1 weights the three classes equally, so the larger class does not dominate model selection. This is a small demonstration, not a claim of production readiness.

## Reproducible result

Results produced from `data/wine.csv` with the repository's current `src/ml.py`:

| Model | Mean CV macro F1 | CV standard deviation | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Most-frequent baseline | 0.1899 | 0.0032 | 0.4000 | 0.1905 |
| Logistic regression | 0.9701 | 0.0277 | 1.0000 | 1.0000 |
| Random forest (selected) | **0.9854** | 0.0206 | 1.0000 | 1.0000 |

The winning forest ranks `flavanoids`, `proline`, `color_intensity`, `alcohol`, and `hue` highest in permutation importance on the held-out rows. This measures how much predictive score changes when a feature is shuffled; it is **not** evidence that a chemical property causes the category.

## What the numbers do and do not show

The perfect result on only 45 test samples is fragile. It says that these models separated this particular test split well. Before using such a model outside a portfolio demo, collect an independent external sample, check stability across collection periods and instruments, document intended use, and monitor errors by group. We did not tune hyperparameters or claim a causal interpretation.

## Reproduce

```bash
pip install -r requirements.txt
python - <<'PY'
import pandas as pd
from src.ml import train
result = train(pd.read_csv('data/wine.csv'), 'wine_class', 'classification')
print(result['scores'].to_string(index=False))
PY
```
