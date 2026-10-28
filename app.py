import streamlit as st
import pnet_model as m

st.set_page_config(page_title="Functioning pNET risk calculator", page_icon="🩺", layout="centered")
st.title("Functioning pNET – high-risk phenotype calculator")
st.caption("Anomaly-detection model (Isolation Forest) derived from 97 resected functioning pancreatic "
           "neuroendocrine tumors (SEER, 2000–2023) and externally validated in an institutional cohort "
           "(IRCCS Azienda Ospedaliero-Universitaria di Bologna, resected 2000 onward). Ricci C, et al.")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age at diagnosis (years)", 18, 95, 55)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        histotype = st.selectbox("Histotype", ["Insulinoma", "Gastrinoma", "Glucagonoma", "VIPoma", "Somatostatinoma", "ACTH-producing"])
        grade = st.radio("WHO grade", ["G1", "G2", "G3"], horizontal=True)
        stage = st.selectbox("Stage", ["Localized", "Regional", "Distant"],
                             help="Localized: confined to the pancreas (N0 M0). Regional: nodal metastases or extension beyond the pancreas. Distant: distant metastases (M1).")
    with c2:
        resection = st.selectbox("Type of resection", ["Enucleation/local excision", "Partial (distal/central) pancreatectomy",
                                                       "Pancreaticoduodenectomy", "Total/extended pancreatectomy"])
        nodes_assessed = st.checkbox("Lymph nodes examined", value=True)
        n_exam = st.number_input("Number of examined lymph nodes", 1, 100, 12, disabled=not nodes_assessed)
        n_pos = st.number_input("Number of positive lymph nodes", 0, 100, 0, disabled=not nodes_assessed)
        rural = st.checkbox("Treated in a non-metropolitan (rural) hospital", value=False)
    go = st.form_submit_button("Calculate", type="primary")

if go:
    if nodes_assessed and n_pos > n_exam:
        st.error("Positive lymph nodes cannot exceed examined lymph nodes.")
        st.stop()
    X = m.build_features(age, sex, histotype, grade, stage, resection, nodes_assessed, n_exam, n_pos, rural)
    s = float(m.anomaly_score(X)[0]); p = float(m.probability(s)); pct = m.percentile(s)
    group = "Low" if p < 0.05 else ("Intermediate" if p <= 0.20 else "High")
    colour = {"Low": "green", "Intermediate": "orange", "High": "red"}[group]

    st.subheader("Result")
    a, b, c = st.columns(3)
    a.metric("Anomaly score", f"{s:.3f}")
    b.metric("Percentile in SEER cohort", f"{pct:.0f}th")
    c.metric("Predicted cancer-specific death", f"{100*p:.1f}%")
    st.markdown(f"### Risk group: :{colour}[{group}]")

    st.markdown("**Observed outcomes by risk group**")
    st.table({
        "Risk group": ["Low (<5%)", "Intermediate (5–20%)", "High (>20%)"],
        "Cancer-specific death, SEER (n = 97)": ["2.0% (1/49)", "8.0% (2/25)", "39.1% (9/23)"],
        "Recurrence/persistence, validation (n = 33)": ["11% (2/19)", "60% (3/5)", "56% (5/9)"],
    })
    st.info("Derived for cancer-specific death (SEER) and externally validated for recurrence or persistent disease "
            "(patients resected from 2000 onward). Recurrences of pNETs may occur many years after resection: "
            "long-term surveillance remains advisable in all risk groups.")

st.divider()
st.caption("For research and educational use only. Not a substitute for clinical judgement. "
           "Higher anomaly scores indicate clinicopathologic profiles that deviate from those of long-term survivors.")
