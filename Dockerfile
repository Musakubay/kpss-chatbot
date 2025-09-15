# Python tabanlı bir imaj seçiyoruz
FROM python:3.13-slim

# Sistem güncellemesi ve tesseract kurulumu
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY . .

# Uygulamayı çalıştır
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]