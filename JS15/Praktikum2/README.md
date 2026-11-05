# Praktikum 2 JS15 - Deployment Model Day vs Night

Implementasi deployment aplikasi web berbasis Flask dan Docker Container untuk studi kasus klasifikasi citra Day vs Night (Siang vs Malam).

## Komponen Berkas
- `app.py`: Web server Flask dengan ekstraksi fitur HOG (Histogram of Oriented Gradients).
- `day_night_model.h5`: Model Neural Network Keras Sequential biner (Sigmoid).
- `scaler.pkl`: StandardScaler 34.596 fitur HOG.
- `Dockerfile`: Konfigurasi container Docker Linux `python:3.9-slim` dengan port 7860.
- `requirements.txt`: Dependensi sistem.

## Menjalankan Secara Lokal
```bash
python app.py
```
Akses di browser pada: `http://localhost:7860`.
