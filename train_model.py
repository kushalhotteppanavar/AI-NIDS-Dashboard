"""Download CIC-IDS2017 CSVs, train AI-NIDS Random Forest, and save dashboard assets."""
from pathlib import Path
import json, urllib.request, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
ARTIFACT_DIR = ROOT / 'artifacts'
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
URLS = {
    'friday_morning.csv': 'https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Morning.pcap_ISCX.csv?download=true',
    'friday_ddos.csv': 'https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv?download=true',
    'friday_portscan.csv': 'https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv?download=true',
}

def download_files():
    for filename, url in URLS.items():
        dest = DATA_DIR / filename
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f'Using existing {filename} ({dest.stat().st_size/1e6:.1f} MB)')
            continue
        print(f'Downloading {filename} ...')
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=120) as response, open(dest, 'wb') as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                out.write(chunk)
        print(f'Saved {filename} ({dest.stat().st_size/1e6:.1f} MB)')

def load_clean(path):
    df = pd.read_csv(path, encoding='cp1252', low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    labels = [c for c in df.columns if c.lower() == 'label']
    if not labels: raise ValueError(f'No Label column found in {path.name}')
    if labels[0] != 'Label': df.rename(columns={labels[0]: 'Label'}, inplace=True)
    return df

def main():
    download_files()
    print('\nReading dataset files ...')
    morning = load_clean(DATA_DIR / 'friday_morning.csv')
    ddos = load_clean(DATA_DIR / 'friday_ddos.csv')
    portscan = load_clean(DATA_DIR / 'friday_portscan.csv')
    benign = morning[morning['Label'].astype(str).str.upper().eq('BENIGN')]
    ddos_rows = ddos[ddos['Label'].astype(str).str.contains('DDoS', case=False, na=False)]
    port_rows = portscan[portscan['Label'].astype(str).str.contains('PortScan', case=False, na=False)]
    n = 10000
    if min(len(benign), len(ddos_rows), len(port_rows)) == 0:
        raise ValueError('One or more expected classes are empty. Check downloaded CSV files and their Label values.')
    parts = [
        benign.sample(min(n, len(benign)), random_state=42),
        ddos_rows.sample(min(n, len(ddos_rows)), random_state=42),
        port_rows.sample(min(n, len(port_rows)), random_state=42),
    ]
    data = pd.concat(parts, ignore_index=True)
    data['Class'] = data['Label'].apply(lambda x: 'BENIGN' if str(x).upper() == 'BENIGN' else ('DDoS' if 'ddos' in str(x).lower() else 'PortScan'))
    y = data['Class'].copy()
    drop = [c for c in ['Flow ID','Source IP','Src IP','Destination IP','Dst IP','Timestamp','Label','Class'] if c in data.columns]
    X = data.drop(columns=drop, errors='ignore').copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    constant = X.columns[X.nunique() <= 1].tolist()
    X.drop(columns=constant, inplace=True)
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    highly_correlated = [c for c in upper.columns if any(upper[c] > 0.98)]
    X.drop(columns=highly_correlated, inplace=True)
    # Keep feature names/order so the dashboard applies the same input schema.
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f'Training Random Forest on {len(X_train):,} rows and {X.shape[1]} features ...')
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = ['BENIGN', 'DDoS', 'PortScan']
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision_weighted': float(precision_score(y_test, pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, pred, average='weighted', zero_division=0)),
        'classes': labels,
        'confusion_matrix': confusion_matrix(y_test, pred, labels=labels).tolist(),
        'classification_report': classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
        'rows_used': int(len(data)),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'feature_count': int(X.shape[1]),
        'class_counts': {str(k): int(v) for k, v in data['Class'].value_counts().items()},
    }
    bundle = {'model': model, 'feature_columns': list(X.columns), 'class_labels': labels,
              'feature_importances': {str(k): float(v) for k,v in zip(X.columns, model.feature_importances_)}}
    joblib.dump(bundle, ARTIFACT_DIR / 'ai_nids_model.joblib')
    (ARTIFACT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    # Save an actual held-out sample and predictions for replay in the dashboard.
    replay = X_test.sample(min(300, len(X_test)), random_state=7).copy()
    replay['actual_class'] = y_test.loc[replay.index].astype(str)
    replay['predicted_class'] = model.predict(replay[X.columns]).astype(str)
    replay['confidence'] = model.predict_proba(replay[X.columns]).max(axis=1)
    replay.insert(0, 'record_id', [f'CIC-{i:04d}' for i in range(1, len(replay)+1)])
    replay.to_csv(ARTIFACT_DIR / 'replay_events.csv', index=False)
    print('\nTRAINING COMPLETE')
    print(json.dumps({k: v for k,v in metrics.items() if k not in ('classification_report','confusion_matrix')}, indent=2))
    print(f'Artifacts saved to: {ARTIFACT_DIR}')

if __name__ == '__main__':
    main()
