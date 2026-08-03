FROM python:3.12-slim

# libgomp1 is needed by PyTorch (OpenMP) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU wheel from its dedicated index (much smaller than the CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY driver_state ./driver_state
COPY dashboard ./dashboard
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh

ENV DS_DATA_ROOT=/app/data \
    DS_MODEL_ROOT=/app/models \
    DS_N_DRIVERS=60 \
    DS_EPOCHS=6 \
    PYTHONUNBUFFERED=1

# Train the models + demo sessions at build time so the container serves instantly
RUN python -m driver_state.cli train

EXPOSE 8000
ENTRYPOINT ["./docker/entrypoint.sh"]
