# Tugas JS15 - Deployment Model Proyek (Klasifikasi BISINDO)

Implementasi pipeline deployment web berbasis Flask dan Docker Container untuk model Machine Learning Proyek: Pengenalan Bahasa Isyarat Indonesia (BISINDO) alfabet 26 huruf (A-Z).

## Komponen Berkas
- `app.py`: Web server Flask dengan ekstraksi 127 fitur MediaPipe HandLandmarker.
- `bisindo_model.h5`: Model Deep Neural Network (Akurasi test: 96.77%).
- `scaler.pkl`: StandardScaler 127 fitur koordinat geometris.
- `labels.pkl`: Pemetaan 26 kelas alfabet (A-Z).
- `hand_landmarker.task`: Berkas model task detector MediaPipe Hand Landmark.
- `Dockerfile`: Konfigurasi container Docker Linux `python:3.9-slim` dengan port 7860.
- `requirements.txt`: Dependensi sistem.

## Menjalankan Secara Lokal
```bash
python app.py
```
Akses di browser pada: `http://localhost:7860`.
