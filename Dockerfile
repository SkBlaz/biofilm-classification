FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/microics:/opt/microics/src \
    MPLBACKEND=Agg \
    MICROICS_DATA_ROOT=/data

WORKDIR /opt/microics

RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        imagemagick \
        libhdf5-dev \
        locales \
        parallel \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.docker.txt /tmp/requirements.docker.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.docker.txt

COPY . /opt/microics

# Copy test datafiles to src directory for CI compatibility
RUN cp /opt/microics/ci_datafile.tsv /opt/microics/src/ && \
    cp /opt/microics/datafile.tsv /opt/microics/src/

RUN mkdir -p /data/jobs \
    && chmod +x /opt/microics/src/run_analysis.sh /opt/microics/src/remove_layers.sh

VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)" || exit 1

CMD ["python", "gui/app.py", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
