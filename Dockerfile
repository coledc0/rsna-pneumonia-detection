FROM python:3.11-slim

WORKDIR /app

# system dpeendencies for opencv and pydicom
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY configs/ configs/

# weights are mounted at runtime, not baked into the image
# (keeps image size small and allows for swapping models without rebuilding)

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]