FROM python:3.12-slim

LABEL org.opencontainers.image.title="MELD Orchestrator"
LABEL org.opencontainers.image.description="MELD orchestrator with FL training support."

WORKDIR /app

RUN mkdir -p /logs /jobs /adapter_store

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY *.py /app/
COPY InternalDataLoader /app/InternalDataLoader
COPY ModelEnvironment /app/ModelEnvironment
COPY ModelManager /app/ModelManager
COPY utils /app/utils
COPY Logger /app/Logger

# Default: single-site inference CLI. For FL: override CMD to ["serve"]
ENTRYPOINT ["python", "main.py"]
CMD ["run"]
