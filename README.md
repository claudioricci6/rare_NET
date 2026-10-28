# Functioning pNET – high-risk phenotype calculator (Streamlit)

Files
- `app.py` – Streamlit interface
- `pnet_model.py` – pure-numpy implementation of the SEER-derived Isolation Forest (identical scores to scikit-learn)
- `model_params.npz`, `model_meta.json` – model parameters (500 trees, scaler, calibration, score percentiles). No patient-level data.
- `requirements.txt`

Run locally
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload all files of this folder (keep them in the same folder).
2. Go to share.streamlit.io → "Create app" → select the repository, branch and `app.py` as main file.
3. Deploy; copy the resulting URL into the manuscript (Discussion, "[URL to be inserted]").
