import re
import joblib
import requests
import streamlit as st
from bs4 import BeautifulSoup


MODEL_PATH = "fake_job_model.joblib"


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_text_from_url(url: str, timeout=15) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove non-text elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # Collect text from common content tags
    parts = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = tag.get_text(" ", strip=True)
        if txt and len(txt) > 20:
            parts.append(txt)

    text = " ".join(parts)
    return clean_text(text)


@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], float(artifact["threshold"])


st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("Fake Job Posting Detector")
st.write("Paste a job posting URL. The app will extract the text and predict Legit vs Fraud.")

pipeline, threshold = load_artifact()

url = st.text_input("Job Post URL", placeholder="https://example.com/job-posting")
manual_text = st.text_area(
    "Or paste job text here (recommended if the site blocks scraping)",
    height=200
)

analyze = st.button("Analyze")

if analyze:
    text = ""

    # Prefer URL if provided; fallback to manual text
    if url.strip():
        try:
            with st.spinner("Fetching and extracting text from the URL..."):
                text = extract_text_from_url(url.strip())
        except Exception as e:
            st.error(f"Could not extract text from URL. Reason: {e}")
            st.info("Tip: Some sites block scraping. Paste the job text manually below and analyze again.")

    if (not text) and manual_text.strip():
        text = clean_text(manual_text.strip())

    if not text:
        st.warning("No text to analyze. Please provide a URL or paste the job text.")
    else:
        proba_fraud = float(pipeline.predict_proba([text])[0][1])
        pred = int(proba_fraud >= threshold)

        st.subheader("Result")
        st.write(f"Fraud probability: **{proba_fraud:.3f}**")
        st.write(f"Decision threshold (high precision): **{threshold:.3f}**")

        if pred == 1:
            st.error("Prediction: **Potentially Fraudulent**")
        else:
            st.success("Prediction: **Likely Legitimate**")

        with st.expander("Show extracted/used text (preview)"):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

        # st.caption("Disclaimer: This is an ML-based estimate for a college project, not a definitive judgement.")