FROM python:3.10-slim

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_VERSION=3.6.2
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy and install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY run.py .
COPY app.py .
COPY setup.py .
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY scripts/ ./scripts/
COPY config.example.json .

# Copy data directory structure (without large files)
RUN mkdir -p data/aoi data/baselines data/dt data/exports \
    data/models data/runs data/satellite_images \
    data/shorelines data/transects

EXPOSE 5000

CMD ["python", "run.py"]
