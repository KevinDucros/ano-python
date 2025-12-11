FROM python:3.11-slim

# Poppler pour pdf2image, Tesseract + langues FRA/ENG
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# FastAPI servie par uvicorn sur le port 3000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]