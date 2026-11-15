from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
MODEL_PATH = ART / 'ai_nids_model.joblib'
METRICS_PATH = ART / 'metrics.json'
REPLAY_PATH = ART / 'replay_events.csv'

st.set_page_config(page_title='AI-NIDS | Threat Monitor', page_icon='🛡️', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1020; color: #e5e7eb;}
[data-testid="stSidebar"] {background: #111827;}
.block-container {padding-top: 1.4rem;}
.kicker {color:#94a3b8; font-size:0.82rem; letter-spacing:.12em; text-transform:uppercase;}
.hero {padding: 1.2rem 1.4rem; border:1px solid #25324a; border-radius:16px; background:linear-gradient(120deg,#111c32,#111827); margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2rem; color:#f8fafc;}
.hero p {color:#a8b5ca; margin:.4rem 0 0 0;}
div[data-testid="stMetric"] {background:#111827; border:1px solid #263247; padding:14px; border-radius:12px;}
.alert {padding:1rem; border-radius:12px; font-weight:700; margin:.4rem 0;}
.good {background:#123329; border:1px solid #1f8a62; color:#86efac;}
.bad {background:#3b1722; border:1px solid #b91c1c; color:#fca5a5;}
.small {color:#94a3b8; font-size:.85rem;}
</style>''', unsafe_allow_html=True)

st.sidebar.title('🛡️ AI-NIDS')
st.sidebar.caption('Network Intrusion Detection System')
page = st.sidebar.radio('Navigation', ['Overview', 'Traffic Replay', 'Inspect a CSV', 'Model & Method'])
st.sidebar.divider()
st.sidebar.markdown('**Dataset**  \nCIC-IDS2017 labelled flow records')
st.sidebar.markdown('**Model**  \nRandom Forest')
st.sidebar.markdown('<p class="small">This demo analyses labelled dataset records. It does not capture live packets from your network.</p>', unsafe_allow_html=True)

if not (MODEL_PATH.exists() and METRICS_PATH.exists() and REPLAY_PATH.exists()):
    st.markdown('<div class="hero"><div class="kicker">Network security analytics</div><h1>AI-Based Intrusion Detection</h1><p>Train the model once to create the dashboard artifacts.</p></div>', unsafe_allow_html=True)
    st.error('Model files are not present yet. First run `python train_model.py` in the VS Code terminal. It downloads CIC-IDS2017 CSV files and trains the model.')
    st.code('python -m venv .venv\n# Windows PowerShell\n.venv\\Scripts\\Activate.ps1\npip install -r requirements.txt\npython train_model.py\nstreamlit run app.py', language='powershell')
    st.stop()

@st.cache_resource
def load_bundle(): return joblib.load(MODEL_PATH)
@st.cache_data
def load_metrics(): return json.loads(METRICS_PATH.read_text(encoding='utf-8'))
@st.cache_data
def load_replay(): return pd.read_csv(REPLAY_PATH)
bundle, metrics, events = load_bundle(), load_metrics(), load_replay()
model, features = bundle['model'], bundle['feature_columns']

if page == 'Overview':
    st.markdown('<div class="hero"><div class="kicker">Security operations • ML monitoring</div><h1>AI-Based Network Intrusion Detection</h1><p>Detect normal traffic, DDoS and PortScan patterns using a trained Random Forest classifier.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Model accuracy', f"{metrics['accuracy']*100:.2f}%")
    c2.metric('Weighted precision', f"{metrics['precision_weighted']*100:.2f}%")
    c3.metric('Weighted recall', f"{metrics['recall_weighted']*100:.2f}%")
    c4.metric('Weighted F1-score', f"{metrics['f1_weighted']*100:.2f}%")
    st.subheader('Detection summary')
    left,right = st.columns([1,1])
    with left:
        counts = pd.Series(metrics['class_counts']).reindex(['BENIGN','DDoS','PortScan']).fillna(0)
        st.markdown('**Training sample by class**')
        st.bar_chart(counts)
    with right:
        st.markdown('**Confusion matrix**')
        cm = pd.DataFrame(metrics['confusion_matrix'], index=metrics['classes'], columns=metrics['classes'])
        st.dataframe(cm, use_container_width=True)
    st.subheader('Latest replay events')
    latest = events.head(12).copy()
    latest['confidence'] = (latest['confidence']*100).round(1).astype(str)+'%'
    st.dataframe(latest[['record_id','actual_class','predicted_class','confidence']], use_container_width=True, hide_index=True)
    st.caption(f"Dataset subset: {metrics['rows_used']:,} records • {metrics['feature_count']} selected features • {metrics['train_rows']:,} train / {metrics['test_rows']:,} test records.")

elif page == 'Traffic Replay':
    st.markdown('<div class="hero"><div class="kicker">Replay console</div><h1>Traffic Replay & Security Alerts</h1><p>Step through real held-out CIC-IDS2017 flow records and inspect the model prediction.</p></div>', unsafe_allow_html=True)
    if 'event_index' not in st.session_state: st.session_state.event_index = 0
    total = len(events)
    a,b,c = st.columns([1,1,2])
    with a:
        if st.button('⟲ Reset replay', use_container_width=True): st.session_state.event_index = 0
    with b:
        if st.button('Next traffic record →', type='primary', use_container_width=True): st.session_state.event_index = (st.session_state.event_index + 1) % total
    with c: st.progress((st.session_state.event_index+1)/total, text=f"Record {st.session_state.event_index+1} of {total}")
    row = events.iloc[st.session_state.event_index]
    malicious = str(row['predicted_class']) != 'BENIGN'
    if malicious:
        st.markdown(f"<div class='alert bad'>🚨 SECURITY ALERT — {row['predicted_class']} predicted</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert good'>✓ NORMAL TRAFFIC — no malicious class predicted</div>", unsafe_allow_html=True)
    x1,x2,x3 = st.columns(3)
    x1.metric('Record ID', str(row['record_id']))
    x2.metric('Predicted class', str(row['predicted_class']))
    x3.metric('Model confidence', f"{float(row['confidence'])*100:.2f}%")
    st.write(f"**Dataset label:** {row['actual_class']}  ·  **Model result:** {row['predicted_class']}")
    if row['actual_class'] != row['predicted_class']:
        st.warning('This record is a misclassification: the model prediction differs from the dataset label.')
    with st.expander('Show selected network features for this record'):
        vals = pd.DataFrame({'Feature': features, 'Value': [row.get(f, np.nan) for f in features]})
        st.dataframe(vals, use_container_width=True, hide_index=True)
    st.subheader('Replay event log')
    log = events.iloc[:st.session_state.event_index+1][['record_id','actual_class','predicted_class','confidence']].copy()
    log['status'] = np.where(log['predicted_class'].eq('BENIGN'), 'NORMAL', 'ALERT')
    log['confidence'] = (log['confidence']*100).round(2).astype(str)+'%'
    st.dataframe(log.tail(25).iloc[::-1], use_container_width=True, hide_index=True)

elif page == 'Inspect a CSV':
    st.markdown('<div class="hero"><div class="kicker">Manual inspection</div><h1>Inspect Network Flow CSV</h1><p>Upload a CSV with the same CIC-IDS2017 feature columns to classify its rows.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader('Upload a network-flow CSV', type=['csv'])
    st.info('The CSV should contain the numeric feature columns used by this model. Missing selected columns are filled with 0; extra columns are ignored. This is intended for compatible flow-feature CSVs, not raw PCAP files.')
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
            raw.columns = [str(c).strip() for c in raw.columns]
            X = raw.reindex(columns=features, fill_value=0).copy()
            for col in features: X[col] = pd.to_numeric(X[col], errors='coerce')
            X.replace([np.inf,-np.inf], np.nan, inplace=True)
            X = X.fillna(0).astype(np.float32)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X).max(axis=1)
            output = raw.copy()
            output['AI_NIDS_Prediction'] = predictions
            output['Confidence'] = np.round(probabilities, 4)
            output['Security_Status'] = np.where(output['AI_NIDS_Prediction'].eq('BENIGN'), 'NORMAL', 'ALERT')
            alert_count = int((predictions != 'BENIGN').sum())
            q1,q2,q3 = st.columns(3)
            q1.metric('Rows inspected', f'{len(output):,}')
            q2.metric('Potential alerts', f'{alert_count:,}')
            q3.metric('Normal predictions', f'{len(output)-alert_count:,}')
            st.dataframe(output.head(500), use_container_width=True, hide_index=True)
            st.download_button('Download predictions as CSV', output.to_csv(index=False).encode('utf-8'), 'ai_nids_predictions.csv', 'text/csv')
        except Exception as e:
            st.error(f'Could not process this CSV: {e}')

else:
    st.markdown('<div class="hero"><div class="kicker">About the implementation</div><h1>Model & Method</h1><p>What the prototype does—and what it does not yet do.</p></div>', unsafe_allow_html=True)
    st.markdown('''**Pipeline**  
1. Load CIC-IDS2017 CSVs.  
2. Select BENIGN, DDoS and PortScan records.  
3. Convert numeric features, handle missing/infinite values, remove constant and highly correlated features.  
4. Split into train/test sets and train a Random Forest.  
5. Evaluate using accuracy, weighted precision/recall/F1 and confusion matrix.  
6. Display predictions and security alerts in the dashboard.''')
    st.subheader('Most important model features')
    imp = pd.Series(bundle['feature_importances']).sort_values(ascending=False).head(15).sort_values()
    st.bar_chart(imp)
    st.subheader('Class metrics')
    report = metrics['classification_report']
    rows=[]
    for label in metrics['classes']:
        r=report.get(label,{})
        rows.append({'Class':label,'Precision':r.get('precision',0),'Recall':r.get('recall',0),'F1-score':r.get('f1-score',0),'Support':int(r.get('support',0))})
    st.dataframe(pd.DataFrame(rows).set_index('Class').style.format({'Precision':'{:.3f}','Recall':'{:.3f}','F1-score':'{:.3f}'}), use_container_width=True)
    st.warning('Scope limitation: the model predicts only classes represented in its training data. This demo replays records from a real labelled dataset; it does not currently capture live packets or guarantee detection of unseen/zero-day attacks.')
