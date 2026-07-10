# =============================================================================
# Runtime container image — one image, two modes (MELD_MODE=train | inference).
#
# Build (from the project root):
#   docker build -t meld-runtime:fl ./runtime
#
# The /artifact folder must contain the Step-1 outputs before building:
#   model.pt            initial backbone weights
#   preprocessor.pkl    Preprocessor fitted on the Step-1 training split
#   model_config.json   backbone architecture + LoRA settings
# Copy them in with:  cp <step1>/results/step1/artifact/* runtime/artifact/
# =============================================================================
FROM python:3.12-slim

LABEL org.opencontainers.image.title="MELD FL runtime"
LABEL org.opencontainers.image.description="Training/inference runtime for the FL simulation (PyTorch MLP/Transformer + LoRA)."

WORKDIR /app

# CPU-only torch keeps the image small; the models are tiny.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pandas numpy scikit-learn pyyaml

# Model code + entrypoint.
COPY entry.py model.py dwh_dataset.py /app/

# Step-1 artifact (backbone, preprocessor, config) baked into the image.
COPY artifact/ /artifact/

# /input and /output are provided by the orchestrator via the shared volume.
# /input and /output are populated by the orchestrator via the Docker API
# (put_archive / get_archive). They must exist as directories in the image.
RUN mkdir -p /input /output
ENTRYPOINT ["python", "entry.py"]
