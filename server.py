"""Small same-origin HTTP app for the DataMind AI portfolio demo."""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from threading import Lock, BoundedSemaphore
from urllib.parse import urlparse
import json
import os
import secrets

import pandas as pd

from src.ml import train, importance, predict_csv


ROOT = Path(__file__).resolve().parent
SAMPLES = {
    'wine': {'name': 'Wine varieties', 'file': 'wine.csv', 'task': 'classification', 'target': 'wine_class', 'description': 'Identify one of three wine classes using 13 chemical measurements.'},
    'iris': {'name': 'Iris flowers', 'file': 'iris.csv', 'task': 'classification', 'target': 'species', 'description': 'Classify flowers by petal and sepal measurements.'},
    'diabetes': {'name': 'Diabetes progression', 'file': 'diabetes.csv', 'task': 'regression', 'target': 'disease_progression', 'description': 'Educational regression example; never use for medical decisions.'},
}
SESSIONS = {}
SESSION_LOCK = Lock()
TRAIN_SLOTS = BoundedSemaphore(1)
MAX_BODY = 12_000_000


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if pd.isna(value):
        return None
    if hasattr(value, 'item'):
        return value.item()
    return value


def dataset(payload):
    if payload.get('sample_id') in SAMPLES and not payload.get('csv'):
        info = SAMPLES[payload['sample_id']]
        return pd.read_csv(ROOT / 'data' / info['file']), info['target'], info['task']
    csv = payload.get('csv')
    if not isinstance(csv, str) or not csv.strip() or len(csv.encode('utf-8')) > 10_000_000:
        raise ValueError('Select a sample or upload a CSV under 10 MB.')
    df = pd.read_csv(StringIO(csv))
    return df, df.columns[-1] if len(df.columns) else '', 'classification'


def inspect(payload):
    df, default_target, default_task = dataset(payload)
    if df.empty or df.shape[1] < 2 or len(df) > 3_000 or df.shape[1] > 40:
        raise ValueError('Use a table with 1–3,000 rows and 2–40 columns.')
    return {'rows': len(df), 'columns': list(df.columns), 'missing': int(df.isna().sum().sum()),
            'preview': json_clean(df.head(6).where(pd.notna(df.head(6)), None).to_dict('records')),
            'default_target': default_target, 'default_task': default_task}


def fit(payload):
    df, default_target, default_task = dataset(payload)
    if df.empty or df.shape[1] < 2 or len(df) > 3_000 or df.shape[1] > 40:
        raise ValueError('Use a table with 1–3,000 rows and 2–40 columns.')
    target = payload.get('target') or default_target
    task = payload.get('task') or default_task
    if target not in df.columns or task not in ('classification', 'regression'):
        raise ValueError('Choose a valid target and task.')
    if not TRAIN_SLOTS.acquire(blocking=False):
        raise ValueError('The demo is busy training other models. Try again shortly.')
    try:
        result = train(df, target, task)
        ranked = result['scores'].round(4).to_dict('records')
        imp = importance(result).head(10).round(4).to_dict('records')
        actual = result['y_test'].head(20).tolist()
        predicted = result['model'].predict(result['x_test'].head(20)).tolist()
        token = secrets.token_urlsafe(24)
        with SESSION_LOCK:
            now = datetime.now(timezone.utc)
            for key, (_, created) in list(SESSIONS.items()):
                if now - created > timedelta(minutes=30):
                    del SESSIONS[key]
            while len(SESSIONS) >= 3:
                del SESSIONS[next(iter(SESSIONS))]
            SESSIONS[token] = result, now
        return {'session_id': token, 'winner': result['winner'], 'scores': json_clean(ranked),
                'importance': json_clean(imp), 'comparison': json_clean([{'actual': a, 'predicted': p} for a, p in zip(actual, predicted)]),
                'features': result['columns'], 'task': task, 'training_rows': len(df) - len(result['x_test']), 'test_rows': len(result['x_test'])}
    finally:
        TRAIN_SLOTS.release()


def predict(payload):
    token = payload.get('session_id')
    with SESSION_LOCK:
        entry = SESSIONS.get(token) if isinstance(token, str) else None
    if not entry or datetime.now(timezone.utc) - entry[1] > timedelta(minutes=30):
        raise ValueError('This model session expired. Train again to predict new rows.')
    csv = payload.get('csv')
    if not isinstance(csv, str) or not csv.strip() or len(csv.encode('utf-8')) > 1_000_000:
        raise ValueError('Upload a prediction CSV under 1 MB.')
    df = pd.read_csv(StringIO(csv))
    if df.empty or len(df) > 1_000:
        raise ValueError('Prediction files must have 1–1,000 rows.')
    output = predict_csv(entry[0], df)
    return {'preview': json_clean(output.head(10).to_dict('records')), 'csv': output.to_csv(index=False), 'rows': len(output)}


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(json_clean(data), ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/health':
            return self.send_json({'status': 'ok'})
        if path == '/api/samples':
            return self.send_json({'samples': [{'id': key, **info} for key, info in SAMPLES.items()]})
        assets = {'/': ('web/index.html', 'text/html; charset=utf-8'), '/app.css': ('web/app.css', 'text/css; charset=utf-8'), '/app.js': ('web/app.js', 'text/javascript; charset=utf-8'), '/favicon.svg': ('web/favicon.svg', 'image/svg+xml')}
        if path not in assets:
            return self.send_json({'error': 'Not found'}, 404)
        file_path, content_type = assets[path]
        self.send_bytes((ROOT / file_path).read_bytes(), content_type)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ('/api/inspect', '/api/train', '/api/predict'):
            return self.send_json({'error': 'Not found'}, 404)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > MAX_BODY:
                return self.send_json({'error': 'Request must be under 12 MB.'}, 413)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Invalid request body.')
            action = {'/api/inspect': inspect, '/api/train': fit, '/api/predict': predict}[path]
            self.send_json(action(payload))
        except (ValueError, KeyError, pd.errors.ParserError, UnicodeError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except Exception:
            self.send_json({'error': 'Unable to process this dataset.'}, 500)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    print(f'DataMind AI listening on {port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
