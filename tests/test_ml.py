import pandas as pd
from sklearn.datasets import load_iris
from src.ml import train, predict_csv


def test_classification_and_new_data():
    data = load_iris(as_frame=True).frame
    data['category'] = ['A', 'B'] * 75
    data.loc[0, 'category'] = None
    result = train(data, 'target', 'classification')
    assert len(result['scores']) == 3
    assert result['scores']['CV F1 (macro)'].between(0, 1).all()
    assert result['scores'].iloc[0]['Model'] == result['winner']
    fresh = data.drop(columns='target').head(2).copy()
    fresh['category'] = 'unseen'
    assert len(predict_csv(result, fresh)) == 2


def test_regression():
    df = pd.DataFrame({'x': range(80), 'label': [float(i * 2) for i in range(80)]})
    result = train(df, 'label', 'regression')
    assert len(result['scores']) == 3
    assert result['scores']['CV MAE'].min() >= 0
