# AI-NIDS Dashboard (CIC-IDS2017)

A local Streamlit dashboard for the AI-Based Real-Time Network Intrusion Detection and Threat Monitoring System project. It trains a Random Forest on a manageable subset of CIC-IDS2017 and displays evaluation metrics, a confusion matrix, feature importance, replay alerts, and CSV classification.

## Requirements
- Windows 10/11, Python 3.10 or 3.11 recommended
- Internet for the first dataset download
- About 2–5 GB of free disk space is a safe allowance for downloaded CSVs, packages and artifacts

## Run in VS Code (Windows PowerShell)
1. Extract the ZIP folder.
2. Open the `AI_NIDS_Dashboard` folder in VS Code.
3. Open Terminal → New Terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

The first `python train_model.py` downloads the three CIC-IDS2017 CSVs and trains the model. It may take several minutes depending on network speed and computer performance. After training, rerun only `streamlit run app.py` to open the dashboard. Streamlit prints a local URL, normally `http://localhost:8501`.

## Dashboard pages
- **Overview:** evaluation metrics, class counts, confusion matrix, recent held-out records.
- **Traffic Replay:** step through held-out CIC-IDS2017 rows and show normal/alert banners.
- **Inspect a CSV:** upload a compatible feature CSV and download predictions.
- **Model & Method:** feature importance, per-class metrics and scope limitations.

## Important accuracy and scope notes
- Dataset traffic is real, labelled CIC-IDS2017 network-flow data.
- Traffic Replay is a replay of held-out dataset records, not live packet capture.
- This model is trained for BENIGN, DDoS and PortScan only. It is not a general-purpose detector for every attack or zero-day activity.
- The uploaded CSV page expects precomputed numeric network-flow features compatible with the training feature schema. It does not convert PCAP files into flows.
- Only use packet capture or testing on networks you own or have explicit permission to monitor.

## Troubleshooting
- If `python` is not recognized, install Python from python.org and enable **Add Python to PATH**, then reopen VS Code.
- If downloads fail, check your internet connection and retry `python train_model.py`. Already downloaded files are reused.
- If training fails due to memory pressure, change `n = 10000` in `train_model.py` to `n = 4000` and rerun.
- If the Streamlit command is not found, activate `.venv` and run `python -m streamlit run app.py`.
