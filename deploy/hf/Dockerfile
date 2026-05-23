# HuggingFace Spaces (Docker SDK) image for the F1 Predictions Streamlit app.
# Inference-only: reads the parquets + reliability PNGs that ship in the
# Space repo. No FastF1 / xgboost / shap at runtime.
FROM python:3.12-slim

# HF Spaces convention: run the app as a non-root uid 1000 user under /home/user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# App code + precomputed artefacts. The page modules use
# Path(__file__).resolve().parents[2] to find the repo root, so keeping the
# layout matching the GitHub source repo lets the same paths resolve here.
COPY --chown=user:user app/ ./app/
COPY --chown=user:user predictions/ ./predictions/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user data/ ./data/

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
