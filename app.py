from pathlib import Path
import json
import threading
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from scapy.all import sniff, conf, IP, TCP, UDP
    SCAPY_READY = True
except Exception:
    SCAPY_READY = False

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
page = st.sidebar.radio('Navigation', ['Overview', 'Live Capture', 'Traffic Replay', 'Inspect a CSV', 'Model & Method'])
st.sidebar.divider()
st.sidebar.markdown('**Dataset**  \nCIC-IDS2017 labelled flow records')
st.sidebar.markdown('**Model**  \nRandom Forest')
st.sidebar.markdown('<p class="small">Supports both dataset replay and continuous live Wi-Fi flow capturing (local Admin).</p>', unsafe_allow_html=True)

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


def get_active_wifi_interface():
    if not SCAPY_READY:
        return None, "Unavailable"
    for iface_obj in conf.ifaces.values():
        name = str(getattr(iface_obj, "name", ""))
        desc = str(getattr(iface_obj, "description", ""))
        ip = str(getattr(iface_obj, "ip", ""))
        if "wi-fi" in name.lower() or "wireless" in desc.lower() or "wi-fi" in desc.lower():
            if ip and not ip.startswith("127.") and not ip.startswith("169.254.") and ip != "0.0.0.0":
                return iface_obj, f"{name} ({ip})"
    default_iface = conf.iface
    return default_iface, f"{getattr(default_iface, 'name', 'Default')} ({getattr(default_iface, 'ip', 'Auto')})"


@st.cache_resource
def get_live_state():
    return {
        "running": False,
        "thread": None,
        "lock": threading.Lock(),
        "packets_total": 0,
        "logs": [],
        "error": None,
        "iface_label": "Detecting...",
    }


def _classify_batch(flows):
    if not flows:
        return []
    feature_rows, meta_rows = [], []
    for (src, dst, sp, dp, pr), f in flows.items():
        dur_sec = max(f["last"] - f["start"], 1e-6)
        dur_us = dur_sec * 1e6
        fwd_lens, bwd_lens = f["fwd_lens"], f["bwd_lens"]
        all_lens = fwd_lens + bwd_lens
        all_times = sorted(f["fwd_times"] + f["bwd_times"])

        iats = np.diff(all_times) * 1e6 if len(all_times) > 1 else np.array([0.0])
        fwd_iats = np.diff(f["fwd_times"]) * 1e6 if len(f["fwd_times"]) > 1 else np.array([0.0])
        bwd_iats = np.diff(f["bwd_times"]) * 1e6 if len(f["bwd_times"]) > 1 else np.array([0.0])

        tot_fwd, tot_bwd = len(fwd_lens), len(bwd_lens)
        sum_fwd, sum_bwd = sum(fwd_lens), sum(bwd_lens)
        tot_bytes, tot_pkts = sum_fwd + sum_bwd, tot_fwd + tot_bwd

        feat_map = {
            "Destination Port": dp,
            "Flow Duration": dur_us,
            "Total Fwd Packets": tot_fwd,
            "Total Backward Packets": tot_bwd,
            "Total Length of Fwd Packets": sum_fwd,
            "Total Length of Bwd Packets": sum_bwd,
            "Fwd Packet Length Max": max(fwd_lens) if fwd_lens else 0,
            "Fwd Packet Length Min": min(fwd_lens) if fwd_lens else 0,
            "Fwd Packet Length Mean": float(np.mean(fwd_lens)) if fwd_lens else 0.0,
            "Fwd Packet Length Std": float(np.std(fwd_lens)) if fwd_lens else 0.0,
            "Bwd Packet Length Max": max(bwd_lens) if bwd_lens else 0,
            "Bwd Packet Length Min": min(bwd_lens) if bwd_lens else 0,
            "Bwd Packet Length Mean": float(np.mean(bwd_lens)) if bwd_lens else 0.0,
            "Bwd Packet Length Std": float(np.std(bwd_lens)) if bwd_lens else 0.0,
            "Flow Bytes/s": tot_bytes / dur_sec,
            "Flow Packets/s": tot_pkts / dur_sec,
            "Flow IAT Mean": float(np.mean(iats)),
            "Flow IAT Std": float(np.std(iats)),
            "Flow IAT Max": float(np.max(iats)),
            "Flow IAT Min": float(np.min(iats)),
            "Fwd IAT Total": float(np.sum(fwd_iats)),
            "Fwd IAT Mean": float(np.mean(fwd_iats)),
            "Fwd IAT Std": float(np.std(fwd_iats)),
            "Fwd IAT Max": float(np.max(fwd_iats)),
            "Fwd IAT Min": float(np.min(fwd_iats)),
            "Bwd IAT Total": float(np.sum(bwd_iats)),
            "Bwd IAT Mean": float(np.mean(bwd_iats)),
            "Bwd IAT Std": float(np.std(bwd_iats)),
            "Bwd IAT Max": float(np.max(bwd_iats)),
            "Bwd IAT Min": float(np.min(bwd_iats)),
            "Fwd Header Length": f["fwd_hdr"],
            "Bwd Header Length": f["bwd_hdr"],
            "Fwd Packets/s": tot_fwd / dur_sec,
            "Bwd Packets/s": tot_bwd / dur_sec,
            "Min Packet Length": min(all_lens),
            "Max Packet Length": max(all_lens),
            "Packet Length Mean": float(np.mean(all_lens)),
            "Packet Length Std": float(np.std(all_lens)),
            "Packet Length Variance": float(np.var(all_lens)),
            "FIN Flag Count": f["fin"],
            "SYN Flag Count": f["syn"],
            "RST Flag Count": f["rst"],
            "PSH Flag Count": f["psh"],
            "ACK Flag Count": f["ack"],
            "URG Flag Count": f["urg"],
            "Down/Up Ratio": (tot_bwd / tot_fwd) if tot_fwd > 0 else 0.0,
            "Average Packet Size": float(np.mean(all_lens)),
            "Avg Fwd Segment Size": float(np.mean(fwd_lens)) if fwd_lens else 0.0,
            "Avg Bwd Segment Size": float(np.mean(bwd_lens)) if bwd_lens else 0.0,
            "Fwd Header Length.1": f["fwd_hdr"],
            "Subflow Fwd Packets": tot_fwd,
            "Subflow Fwd Bytes": sum_fwd,
            "Subflow Bwd Packets": tot_bwd,
            "Subflow Bwd Bytes": sum_bwd,
            "Init_Win_bytes_forward": f["init_win_fwd"],
            "Init_Win_bytes_backward": f["init_win_bwd"],
            "act_data_pkt_fwd": tot_fwd,
            "min_seg_size_forward": min(fwd_lens) if fwd_lens else 0,
        }

        feature_rows.append({col: feat_map.get(col.strip(), 0.0) for col in features})
        meta_rows.append({
            "Time": time.strftime("%H:%M:%S"),
            "Source IP": src,
            "Destination IP": dst,
            "Dst Port": dp,
            "Protocol": "TCP" if pr == 6 else ("UDP" if pr == 17 else str(pr)),
            "Packets": tot_pkts,
            "Bytes": tot_bytes,
        })

    X_live = pd.DataFrame(feature_rows)[features].fillna(0).astype(np.float32)
    preds = model.predict(X_live)
    probs = model.predict_proba(X_live).max(axis=1)

    records = []
    for meta, pred, prob in zip(meta_rows, preds, probs):
        meta["predicted_class"] = str(pred)
        meta["confidence"] = f"{prob * 100:.1f}%"
        meta["status"] = "NORMAL" if str(pred) == "BENIGN" else "ALERT"
        records.append(meta)
    return records


def _sniffer_worker(state):
    wifi_iface, iface_label = get_active_wifi_interface()
    with state["lock"]:
        state["iface_label"] = iface_label
        state["error"] = None

    while state["running"]:
        try:
            packets = sniff(timeout=2, iface=wifi_iface, store=True)
            flows = {}
            for pkt in packets:
                if IP not in pkt:
                    continue
                ip = pkt[IP]
                proto = int(ip.proto)
                sport = int(pkt[TCP].sport) if TCP in pkt else (int(pkt[UDP].sport) if UDP in pkt else 0)
                dport = int(pkt[TCP].dport) if TCP in pkt else (int(pkt[UDP].dport) if UDP in pkt else 0)

                fwd_key = (ip.src, ip.dst, sport, dport, proto)
                bwd_key = (ip.dst, ip.src, dport, sport, proto)
                if fwd_key in flows:
                    key, direction = fwd_key, "fwd"
                elif bwd_key in flows:
                    key, direction = bwd_key, "bwd"
                else:
                    key, direction = fwd_key, "fwd"
                    flows[key] = {
                        "start": float(pkt.time), "last": float(pkt.time),
                        "fwd_times": [], "bwd_times": [],
                        "fwd_lens": [], "bwd_lens": [],
                        "fwd_hdr": 0, "bwd_hdr": 0,
                        "syn": 0, "ack": 0, "psh": 0, "rst": 0, "fin": 0, "urg": 0,
                        "init_win_fwd": 0, "init_win_bwd": 0, "dport": dport,
                    }

                f = flows[key]
                t = float(pkt.time)
                pkt_len = len(pkt)
                hdr_len = (ip.ihl * 4) + (pkt[TCP].dataofs * 4 if TCP in pkt else (8 if UDP in pkt else 0))
                f["last"] = max(f["last"], t)

                if direction == "fwd":
                    if not f["fwd_times"] and TCP in pkt:
                        f["init_win_fwd"] = int(pkt[TCP].window)
                    f["fwd_times"].append(t)
                    f["fwd_lens"].append(pkt_len)
                    f["fwd_hdr"] += hdr_len
                else:
                    if not f["bwd_times"] and TCP in pkt:
                        f["init_win_bwd"] = int(pkt[TCP].window)
                    f["bwd_times"].append(t)
                    f["bwd_lens"].append(pkt_len)
                    f["bwd_hdr"] += hdr_len

                if TCP in pkt:
                    flags = pkt[TCP].flags
                    if flags & 0x02: f["syn"] += 1
                    if flags & 0x10: f["ack"] += 1
                    if flags & 0x08: f["psh"] += 1
                    if flags & 0x04: f["rst"] += 1
                    if flags & 0x01: f["fin"] += 1
                    if flags & 0x20: f["urg"] += 1

            new_records = _classify_batch(flows)
            with state["lock"]:
                state["packets_total"] += len(packets)
                if new_records:
                    state["logs"].extend(new_records)
                    state["logs"] = state["logs"][-200:]
        except PermissionError:
            with state["lock"]:
                state["error"] = "Permission denied: Close VS Code, right-click VS Code -> 'Run as administrator', and restart."
                state["running"] = False
            break
        except Exception as exc:
            with state["lock"]:
                state["error"] = f"Sniffer error: {exc}"
                state["running"] = False
            break


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

elif page == 'Live Capture':
    st.markdown('<div class="hero"><div class="kicker">Real-time NIC sensor • Wi-Fi monitor</div><h1>Live Wi-Fi Capture & Threat Monitor</h1><p>Continuously captures live traffic on your active Wi-Fi adapter while you browse, search, or run network tests.</p></div>', unsafe_allow_html=True)

    if not SCAPY_READY:
        st.warning('Scapy/Npcap is not installed or unavailable in this environment. Install Npcap and run `pip install scapy` in VS Code (as Administrator).')
    else:
        state = get_live_state()
        _, detected_wifi = get_active_wifi_interface()

        b1, b2, b3 = st.columns([1, 1, 2])
        with b1:
            if st.button('▶ Start Live Capture', type='primary', use_container_width=True, disabled=state['running']):
                state['running'] = True
                t = threading.Thread(target=_sniffer_worker, args=(state,), daemon=True)
                state['thread'] = t
                t.start()
                st.rerun()
        with b2:
            if st.button('⏹ Stop Capture', use_container_width=True, disabled=not state['running']):
                state['running'] = False
                st.rerun()
        with b3:
            if st.button('⟲ Clear Live Log', use_container_width=True):
                with state['lock']:
                    state['logs'] = []
                    state['packets_total'] = 0
                st.rerun()

        st.caption(f"Active Source Interface: **{detected_wifi}** • Capture Status: **{'🟢 LIVE MONITORING (Switch to browser & search now!)' if state['running'] else '⚪ STOPPED'}**")

        if state['error']:
            st.error(state['error'])

        with state['lock']:
            pkts_total = state['packets_total']
            logs_copy = list(state['logs'])

        live_df = pd.DataFrame(logs_copy)
        alert_flows = live_df[live_df['predicted_class'] != 'BENIGN'] if not live_df.empty else pd.DataFrame()

        if not alert_flows.empty:
            latest_threat = alert_flows.iloc[-1]['predicted_class']
            st.markdown(f"<div class='alert bad'>🚨 LIVE WI-FI SECURITY ALERT — {len(alert_flows)} malicious flow(s) detected (Latest: {latest_threat})</div>", unsafe_allow_html=True)
        elif not live_df.empty:
            st.markdown("<div class='alert good'>✓ LIVE WI-FI TRAFFIC NORMAL — All captured network flows classified as BENIGN</div>", unsafe_allow_html=True)
        else:
            st.info("Click **Start Live Capture**, then open your browser and search anything (or run a local port scan) to see live flows appear here.")

        m1, m2, m3 = st.columns(3)
        m1.metric('Raw Packets Captured', f"{pkts_total:,}")
        m2.metric('Flows Analyzed', f"{len(live_df):,}")
        m3.metric('Threats Detected', f"{len(alert_flows):,}")

        if not live_df.empty:
            st.subheader('Live Wi-Fi Flow Log (Newest First)')
            st.dataframe(live_df.iloc[::-1], use_container_width=True, hide_index=True)

        if state['running']:
            time.sleep(2)
            st.rerun()

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

            # --- ADD THIS FILTER WIDGET ---
            st.markdown("---")
            filter_choice = st.radio(
                "Filter table view:", 
                ["Show All Rows", "Show Only Attacks / Alerts (🚨)", "Show Only Normal (BENIGN)"], 
                horizontal=True
            )

            display_df = output
            if filter_choice == "Show Only Attacks / Alerts (🚨)":
                display_df = output[output['AI_NIDS_Prediction'] != 'BENIGN']
            elif filter_choice == "Show Only Normal (BENIGN)":
                display_df = output[output['AI_NIDS_Prediction'] == 'BENIGN']

            st.write(f"Showing {min(len(display_df), 500):,} of {len(display_df):,} matching rows:")
            st.dataframe(display_df.head(500), use_container_width=True, hide_index=True)
            # ------------------------------

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
