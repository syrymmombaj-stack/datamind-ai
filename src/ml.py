"""Reproducible tabular classification/regression pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def prepare(df, target, task):
    if target not in df or task not in ('classification', 'regression'):
        raise ValueError('Select a valid target and task.')
    clean = df.dropna(subset=[target]).copy()
    if task == 'regression':
        clean[target] = pd.to_numeric(clean[target], errors='coerce')
        clean = clean.dropna(subset=[target])
    x = clean.drop(columns=[target]).replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, x.notna().any()]
    for col in x.select_dtypes(exclude='number'):
        x[col] = x[col].astype('string').fillna('__missing__').astype(str)
    y = clean[target]
    if len(x) < 25 or x.empty or y.nunique() < 2:
        raise ValueError('Need at least 25 labeled rows, one usable feature and two distinct target values.')
    if task == 'classification' and y.nunique() > min(30, len(y) // 4):
        raise ValueError('Too many distinct classes. Choose regression for a numeric continuous target.')
    if task == 'classification' and y.value_counts().min() < 6:
        raise ValueError('Each target class needs at least six examples for stratified cross validation.')
    return x, y


def train(df, target, task):
    x, y = prepare(df, target, task)
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = x.columns.difference(numeric).tolist()
    if any(x[c].nunique() > 100 for c in categorical):
        raise ValueError('A text or ID column has over 100 categories. Remove it before training.')
    transforms = []
    if numeric:
        transforms.append(('numeric', SimpleImputer(strategy='median', add_indicator=True), numeric))
    if categorical:
        transforms.append(('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical))
    preprocess = ColumnTransformer(transforms)
    stratify = y if task == 'classification' else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42, stratify=stratify)
    if task == 'classification':
        candidates = {'Baseline': DummyClassifier(strategy='most_frequent'), 'Logistic regression': LogisticRegression(max_iter=1000), 'Random forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)}
    else:
        candidates = {'Baseline': DummyRegressor(strategy='mean'), 'Ridge regression': Ridge(), 'Random forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)}
    results, fitted = [], {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if task == 'classification' else KFold(n_splits=3, shuffle=True, random_state=42)
    scoring = 'f1_macro' if task == 'classification' else 'neg_mean_absolute_error'
    for name, estimator in candidates.items():
        model = Pipeline([('preprocess', preprocess), ('model', estimator)])
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if task == 'classification':
            metrics = {'CV F1 (macro)': cv_scores.mean(), 'CV std': cv_scores.std(), 'Test accuracy': accuracy_score(y_test, pred), 'Test F1 (macro)': f1_score(y_test, pred, average='macro', zero_division=0)}
        else:
            metrics = {'CV MAE': -cv_scores.mean(), 'CV std': cv_scores.std(), 'Test MAE': mean_absolute_error(y_test, pred), 'Test RMSE': np.sqrt(mean_squared_error(y_test, pred)), 'Test R²': r2_score(y_test, pred)}
        results.append({'Model': name, **metrics})
        fitted[name] = model
    score = 'CV F1 (macro)' if task == 'classification' else 'CV MAE'
    ordered = pd.DataFrame(results).sort_values(score, ascending=(task == 'regression'))
    winner = ordered.iloc[0]['Model']
    return {'scores': ordered, 'model': fitted[winner], 'winner': winner, 'x_test': x_test, 'y_test': y_test, 'columns': x.columns.tolist(), 'task': task, 'target': target}


def importance(result):
    scoring = 'f1_macro' if result['task'] == 'classification' else 'neg_mean_absolute_error'
    sample = result['x_test'].sample(min(300, len(result['x_test'])), random_state=42)
    y = result['y_test'].loc[sample.index]
    scores = permutation_importance(result['model'], sample, y, scoring=scoring, n_repeats=3, random_state=42)
    return pd.DataFrame({'Feature': sample.columns, 'Importance': scores.importances_mean}).sort_values('Importance', ascending=False)


def predict_csv(result, df):
    missing = set(result['columns']) - set(df.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    x = df[result['columns']].copy().replace([np.inf, -np.inf], np.nan)
    for col in x.select_dtypes(exclude='number'):
        x[col] = x[col].astype('string').fillna('__missing__').astype(str)
    output = df.copy()
    output['prediction'] = result['model'].predict(x)
    return output
