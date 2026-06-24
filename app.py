import streamlit as st
import pandas as pd
import tempfile
import os
import time
from pathlib import Path

from src.pipeline import RecruiterPipeline
from config import FAISS_INDEX_PATH, EMBEDDINGS_META_PATH

st.set_page_config(
    page_title="RoleFit AI Recruiter",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RoleFit: AI-Powered Candidate Ranking")
st.markdown("""
Upload a `candidates.jsonl` file to rank them against the Senior AI/ML Engineer Job Description.
*Note: This sandbox runs the exact same pipeline used for the 100K challenge, but uses a pre-computed FAISS index. Please ensure the uploaded candidates were part of the pre-computation.*
""")

# Setup pipeline (cached so we don't reload models on every UI interaction)
@st.cache_resource
def load_pipeline():
    with st.spinner("Loading AI Models (Bi-encoder, Cross-encoder)..."):
        return RecruiterPipeline()

pipeline = load_pipeline()

uploaded_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl"])

if uploaded_file is not None:
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if st.button("🚀 Run Ranking Pipeline"):
        if not FAISS_INDEX_PATH.exists():
            st.error(f"FAISS index missing at {FAISS_INDEX_PATH}. Please run `python precompute.py` first.")
        else:
            with st.spinner("Ranking Candidates... This includes vector retrieval, cross-encoder re-ranking, and rule-based behavioral scoring."):
                start_time = time.time()
                
                # Output to a temp CSV
                out_csv = tmp_path.replace(".jsonl", "_out.csv")
                
                try:
                    rows = pipeline.run(tmp_path, out_csv)
                    elapsed = time.time() - start_time
                    
                    st.success(f"Successfully ranked candidates in {elapsed:.2f}s!")
                    
                    # Display results
                    df = pd.DataFrame([vars(r) for r in rows])
                    st.dataframe(
                        df,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank"),
                            "candidate_id": st.column_config.TextColumn("Candidate ID"),
                            "score": st.column_config.NumberColumn("Composite Score", format="%.4f"),
                            "reasoning": st.column_config.TextColumn("Reasoning")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Provide download link
                    with open(out_csv, "rb") as file:
                        btn = st.download_button(
                            label="📥 Download  CSV",
                            data=file,
                            file_name="submission.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"An error occurred during ranking: {e}")
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
