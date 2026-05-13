set -e

# pip install streamlit plotly --quiet

cd "$REPO_ROOT"
streamlit run "filter_device/app_labelisation.py" \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.runOnSave true
