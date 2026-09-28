from pathlib import Path
import pandas as pd
import streamlit as st
from src.ml import train, importance, predict_csv

st.set_page_config(page_title='DataMind AI', page_icon='🧠', layout='wide')
st.title('🧠 DataMind AI')
st.caption('Explore data, compare real ML models, explain predictions. No paid API required.')
st.sidebar.header('Dataset')
upload = st.sidebar.file_uploader('Upload a CSV (up to 10 MB)', type='csv')
samples = {
    'Wine varieties · classification': ('wine.csv', 'classification', 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html'),
    'Iris flowers · classification': ('iris.csv', 'classification', 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html'),
    'Diabetes progression · regression': ('diabetes.csv', 'regression', 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html'),
}
if upload is None:
    selection = st.sidebar.selectbox('Included dataset', list(samples))
    filename, suggested_task, source_url = samples[selection]
    df = pd.read_csv(Path(__file__).parent / 'data' / filename)
    signature_data = filename.encode()
    st.info(f'Exploring {selection}. [Dataset source]({source_url})')
    st.download_button('Download this sample CSV', df.to_csv(index=False).encode(), filename, 'text/csv')
else:
    suggested_task = 'classification'
    signature_data = upload.getvalue()
    if upload.size > 10_000_000:
        st.error('CSV exceeds the 10 MB limit.')
        st.stop()
    try:
        df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f'Could not read CSV: {exc}')
        st.stop()

if df.empty or len(df.columns) < 2:
    st.error('The file needs rows and at least two columns.')
    st.stop()
st.sidebar.write(f'{len(df):,} rows · {len(df.columns)} columns')
target = st.sidebar.selectbox('Target column', df.columns, index=len(df.columns)-1)
task = st.sidebar.radio('Problem type', ['classification', 'regression'], index=['classification', 'regression'].index(suggested_task), key=f'task_{upload.name if upload else filename}')
overview, modeling, forecasting = st.tabs(['Explore', 'Train & evaluate', 'Predict'])
with overview:
    a, b, c = st.columns(3)
    a.metric('Rows', f'{len(df):,}')
    b.metric('Features', len(df.columns)-1)
    c.metric('Missing cells', int(df.isna().sum().sum()))
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader('Missing values by column')
    st.bar_chart(df.isna().sum().sort_values(ascending=False))
    st.subheader('Numeric summary')
    st.dataframe(df.describe().T if not df.select_dtypes('number').empty else pd.DataFrame(), use_container_width=True)
with modeling:
    st.write('Models are ranked by three-fold cross validation on the training data. A separate 25% test split is held out until final evaluation. Preprocessing is fitted within each fold.')
    if st.button('Train models', type='primary'):
        try:
            with st.spinner('Training and evaluating…'):
                st.session_state['result'] = train(df, target, task)
                st.session_state['signature'] = (signature_data, target, task)
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.get('result')
    signature = (signature_data, target, task)
    if result and st.session_state.get('signature') == signature:
        st.success(f"Best cross validation score: {result['winner']}")
        st.dataframe(result['scores'].style.format(precision=3), use_container_width=True)
        st.caption('CV is used to select the model. Test metrics are measured after selection on untouched rows. Small datasets may still have noisy estimates.')
        with st.expander('Feature importance (permutation)'):
            if st.button('Calculate importance'):
                try:
                    st.bar_chart(importance(result).set_index('Feature'))
                except Exception as exc:
                    st.error(str(exc))
        st.subheader('Predicted vs actual')
        comparison = pd.DataFrame({'actual': result['y_test'], 'predicted': result['model'].predict(result['x_test'])})
        st.dataframe(comparison.head(30), use_container_width=True)
with forecasting:
    if not result or st.session_state.get('signature') != signature:
        st.info('Train models on the current dataset first.')
    else:
        st.write('Upload a CSV with the same feature column names. The target column is optional.')
        new_file = st.file_uploader('New data for prediction', type='csv', key='new')
        if new_file:
            try:
                output = predict_csv(result, pd.read_csv(new_file))
                st.dataframe(output.head(100), use_container_width=True)
                st.download_button('Download predictions', output.to_csv(index=False).encode(), 'predictions.csv', 'text/csv')
            except Exception as exc:
                st.error(str(exc))
