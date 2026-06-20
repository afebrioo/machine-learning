# 🚀 Machine Learning & Midterm Projects Repository

| Identitas | Detail |
| :--- | :--- |
| **Nama** | Rahmanda Afebrio Yuris Soesatyo |
| **NIM** | 1103223024 |

Repositori ini berisi kumpulan tugas perkuliahan, replikasi kode buku referensi, dan proyek ujian tengah semester (UTS) untuk mata kuliah **Machine Learning**. Seluruh kode program, notebook Jupyter, dan visualisasi disajikan dalam **Bahasa Indonesia** secara premium.

---

## 📚 1. Replikasi & Pembelajaran Buku Acuan

Di bawah ini adalah folder pembelajaran untuk masing-masing buku referensi:

### 1.1. [Introduction to Machine Learning with Python](Introduction%20to%20Machine%20Learning%20with%20Python/)
*   **Karya:** Andreas C. Müller & Sarah Guido
*   **Konten:** Replikasi Bab 1 s.d. Bab 8.
*   **Fokus:** Konsep dasar machine learning menggunakan pustaka `scikit-learn` dan visualisasi `mglearn`.
*   **Tautan Folder:** [Pergi ke folder Introduction to Machine Learning dengan Python](Introduction%20to%20Machine%20Learning%20with%20Python/)

### 1.2. [scikit-learn Cookbook, Third Edition](scikit-learn%20Cookbook,%20Third%20Edition/)
*   **Karya:** John Sukup (Packt Publishing)
*   **Konten:** Resep kode praktis Bab 1 s.d. Bab 13 (Notebook utama + Solusi latihan).
*   **Fokus:** Resep praktis machine learning (*Cookbook*) berorientasi industri menggunakan `scikit-learn` v1.5+.
*   **Tautan Folder:** [Pergi ke folder scikit-learn Cookbook, Third Edition](scikit-learn%20Cookbook,%20Third%20Edition/)

### 1.3. [Practical Linear Algebra for Data Science](Practical%20Linear%20Algebra%20for%20Data%20Science/)
*   **Karya:** Mike X. Cohen (O'Reilly)
*   **Konten:** Replikasi Bab 1 s.d. Bab 16 (Notebook utama + Solusi latihan).
*   **Fokus:** Fondasi aljabar linier komputasional untuk data science menggunakan Python, NumPy, SciPy, dan visualisasi grafik.
*   **Tautan Folder:** [Pergi ke folder Practical Linear Algebra for Data Science](Practical%20Linear%20Algebra%20for%20Data%20Science/)

### 1.4. [Practical Statistics for Data Scientists](Practical%20Statistics%20for%20Data%20Scientists/)
*   **Karya:** Peter Bruce, Andrew Bruce, & Peter Gedeck (O'Reilly)
*   **Konten:** Replikasi Bab 1 s.d. Bab 7 (Notebook utama + Dataset lokal).
*   **Fokus:** Konsep statistika dasar dan menengah untuk data science menggunakan Python, Pandas, Statsmodels, wquantiles, dan dmba.
*   **Tautan Folder:** [Pergi ke folder Practical Statistics for Data Scientists](Practical%20Statistics%20for%20Data%20Scientists/)

---

## 🛡️ 2. Proyek Midterm (UTS) Machine Learning

Materi proyek ujian tengah semester yang terdiri dari tiga bagian utama:

### 2.1. Proyek 1: Klasifikasi - Mendeteksi Transaksi Fraud (Imbalanced Data)
*   **Tujuan:** Klasifikasi biner (Fraud vs Non-Fraud) pada transaksi finansial dengan ketidakseimbangan kelas tinggi menggunakan SMOTE dan Multi-Layer Perceptron (MLP).
*   **Folder:** [Midterm-1](Midterm-1/)

### 2.2. Proyek 2: Regresi - Memprediksi Tahun Rilis Lagu (MLP)
*   **Tujuan:** Memprediksi tahun rilis lagu berdasarkan 90 fitur akustik menggunakan regresi MLP.
*   **Folder:** [Midterm-2](Midterm-2/)

### 2.3. Proyek 3: Clustering - Segmentasi Pelanggan Kartu Kredit (DEC)
*   **Tujuan:** Segmentasi pelanggan menggunakan Deep Embedded Clustering (DEC) dengan autoencoder.
*   **Folder:** [Midterm-3](Midterm-3/)

---

## 🎓 3. Proyek UAS (Ujian Akhir Semester) Machine Learning

Proyek ujian akhir semester dengan pipeline Deep Learning end-to-end untuk dua tugas berbeda:

### 3.1. Task 1 — Fraud Detection (Klasifikasi)
*   **Tujuan:** Klasifikasi biner transaksi fraud menggunakan MLP + SMOTE + Optuna hyperparameter tuning.
*   **Dataset:** 590,540 transaksi finansial | Fitur: ~394 kolom setelah preprocessing
*   **Model:** MLP (Binary Crossentropy) + Optuna + MLFlow tracking
*   **Folder:** [UAS/Task1_FraudDetection](UAS/Task1_FraudDetection/)

### 3.2. Task 2 — Song Year Regression (Regresi)
*   **Tujuan:** Memprediksi tahun rilis lagu berdasarkan 90 fitur audio numerik.
*   **Dataset:** 515,345 lagu | Rentang tahun: 1922–2011 | 95 fitur (setelah feature engineering)
*   **Model:** MLP (Huber Loss) + **LightGBM** sebagai pembanding + Optuna + MLFlow + LIME
*   **Hasil Terbaik:** MLP — MAE **5.64 tahun** | R² **0.360**
*   **Folder:** [UAS/Task2_SongYearRegression](UAS/Task2_SongYearRegression/)

---

## 🛠️ Persyaratan Lingkungan (Requirements)

Untuk menjalankan seluruh notebook di repositori ini, pasang dependensi utama berikut:
```bash
pip install numpy<2 pandas matplotlib seaborn scipy sympy scikit-learn scikit-image statsmodels dmba wquantiles tensorflow torch plotly optuna mlflow imbalanced-learn lime lightgbm
```
