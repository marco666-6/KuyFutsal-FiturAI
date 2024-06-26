# Menggunakan base image Python
FROM python:3.10-slim

# Mengekspos port 5000
EXPOSE 5000

# Menjaga Python dari menghasilkan file .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Mematikan buffering untuk logging container yang lebih mudah
ENV PYTHONUNBUFFERED=1

# Menyalin dan menginstal requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Menentukan direktori kerja
WORKDIR /app
COPY . /app

# Menyalin model ke dalam direktori kontainer
COPY DeployModel/Kuysal_chatbot_pipeline.pkl /app/Kuysal_chatbot_pipeline.pkl
COPY DeployModel/intents.json /app/intents.json

# Membuat pengguna non-root dengan UID khusus dan memberikan izin untuk mengakses folder /app
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Menjalankan Gunicorn untuk aplikasi Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "DeployModel.app:app"]