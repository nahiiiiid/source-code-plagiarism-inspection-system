import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import os, subprocess, json, tempfile, zipfile, shlex
from io import StringIO
from src.preprocess import tokenize_code, simple_identifier_normalize, line_level_similarity, tokens_to_string
from src.features import TfidfCodeVectorizer, cosine_sim_from_vectors

# Must be the very first Streamlit command
st.set_page_config(page_title="Source Code Plagiarism Inspection System", layout="wide")

st.title("Source Code Plagiarism Inspection System (Cutting-edge)")
st.markdown("Upload two files for pairwise check, or scan a folder of submissions for batch detection.")

# ---------------- Pairwise Check ----------------
st.header("Pairwise check")
col1, col2 = st.columns(2)
with col1:
    codeA = st.text_area("Code A", height=200)
with col2:
    codeB = st.text_area("Code B", height=200)

backend = st.selectbox("Backend", options=['tfidf','codebert'])

if st.button("Check similarity (pairwise)"):
    if not codeA.strip() or not codeB.strip():
        st.warning("Paste code into both fields or upload files.")
    else:
        # Preprocessing
        tA = simple_identifier_normalize(tokenize_code(codeA))
        tB = simple_identifier_normalize(tokenize_code(codeB))
        sA, sB = tokens_to_string(tA), tokens_to_string(tB)

        # TF-IDF similarity
        vec = TfidfCodeVectorizer()
        vec.fit([sA, sB])
        vA = vec.transform([sA])
        vB = vec.transform([sB])
        cos = cosine_sim_from_vectors(vA, vB)
        st.metric("TF-IDF cosine", f"{cos:.3f}")

        # Optional CodeBERT similarity
        if backend == 'codebert':
            try:
                from src.embeddings import CodeEmbedder
                from sklearn.metrics.pairwise import cosine_similarity
                ce = CodeEmbedder()
                emb = ce.embed([sA, sB])
                emb_cos = cosine_similarity(emb[0:1], emb[1:2])[0,0]
                st.metric("CodeBERT cosine", f"{emb_cos:.3f}")
            except Exception as e:
                st.warning("CodeBERT not available or failed: " + str(e))

        # Line-level matches
        matches = line_level_similarity(codeA, codeB)
        st.subheader("Top line matches (ratio>0.5)")
        import pandas as pd
        match_rows = []
        for i, j, r, a, b in matches[:10]:
            if r > 0.5:
                match_rows.append({"Line A": f"{i+1}: {a}", "Line B": f"{j+1}: {b}", "Similarity": r})
        if match_rows:
            st.dataframe(pd.DataFrame(match_rows))
        else:
            st.info("No line-level matches above 0.5 similarity.")

# ---------------- Batch Scan ----------------
st.header("Batch scan folder")
uploaded = st.file_uploader("Or upload a zip of submissions for batch scan (zip)", type=['zip'])
folder_path = st.text_input("Or local folder path (server)", value="submissions")
threshold = st.slider("Score threshold", 0.5, 0.99, 0.8)
backend_scan = st.selectbox("Backend for scan", options=['tfidf','codebert'], key="scan_backend")

if uploaded is not None:
    tmpdir = tempfile.mkdtemp()
    zpath = os.path.join(tmpdir, "uploads.zip")
    with open(zpath, "wb") as f:
        f.write(uploaded.getbuffer())
    with zipfile.ZipFile(zpath, 'r') as z:
        z.extractall(tmpdir)
    st.success(f"Extracted to {tmpdir}")
    folder_path = tmpdir

if st.button("Run batch scan"):
    if not os.path.exists(folder_path):
        st.error("Folder not found on server. Upload zip or provide server path.")
    else:
        st.info("Running scan... this may take time for large folders.")
        cmd = f'python -m src.batch_scan --folder "{folder_path}" --out report.csv --threshold {threshold} --backend {backend_scan}'
        st.write("Command:", cmd)

        try:
            proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
            st.text(proc.stdout)
            st.text(proc.stderr)
        except Exception as e:
            st.error("Batch scan failed: " + str(e))

        # Display CSV if generated
        if os.path.exists("report.csv"):
            try:
                import pandas as pd
                df = pd.read_csv("report.csv")
                st.success("Report generated: report.csv")
                st.dataframe(df)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download report.csv", csv_bytes, file_name="report.csv", mime="text/csv")
            except Exception as e:
                st.error("Failed to read report.csv: " + str(e))
