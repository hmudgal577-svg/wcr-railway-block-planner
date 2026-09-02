FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for GeoPandas / Shapely / C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port (Render dynamically provides $PORT)
ENV PORT=8501
EXPOSE 8501

# Run Streamlit
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
