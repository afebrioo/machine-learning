# 🎓 UAS Machine Learning — Deep Learning Pipeline

> **Mata Kuliah**: Machine Learning  

---

## 👤 Identitas Mahasiswa

| Field | Detail |
|-------|--------|
| **Nama** | Rahmanda Afebrio Yuris Soesatyo |
| **NIM** | 1103223024 |

---

## 📌 Deskripsi Repository

Repository ini berisi implementasi pipeline **Deep Learning end-to-end** untuk dua tugas berbeda:

1. **Task 1 — Fraud Detection**: Klasifikasi biner untuk mendeteksi transaksi online yang bersifat fraudulent menggunakan dataset transaksi finansial.
2. **Task 2 — Song Year Regression**: Regresi untuk memprediksi tahun rilis lagu berdasarkan fitur audio numerik.

Setiap pipeline mencakup:
- ✅ Eksplorasi dan Analisis Data (EDA)
- ✅ Preprocessing & Feature Engineering
- ✅ Model Deep Learning (MLP dengan Keras/TensorFlow)
- ✅ Hyperparameter Tuning dengan **Optuna**
- ✅ Evaluasi menggunakan metrik yang sesuai
- ✅ Tracking eksperimen dengan **MLFlow**
- ✅ Interpretasi model dengan **LIME** (Task 2)

---

## 📁 Struktur Repository

```
UAS/
│
├── Task1_FraudDetection/
│   ├── Task1_FraudDetection.ipynb   ← Notebook utama Task 1
│   └── train_transaction.csv        ← Dataset (letakkan di sini)
│
├── Task2_SongYearRegression/
│   ├── Task2_SongYearRegression.ipynb  ← Notebook utama Task 2
│   └── midterm-regresi-dataset.csv     ← Dataset (letakkan di sini)
│
└── README.md                        ← Dokumen ini
```

---

## 📊 Task 1: Fraud Detection (Klasifikasi)

### Dataset
- **File**: `train_transaction.csv`
- **Sumber**: [Google Drive](https://drive.google.com/file/d/1If2Xpro_dx8IfNkqDnw3qZ85v_eKtB6B/view)
- **Deskripsi**: Data transaksi online dengan label biner `isFraud` (0 = aman, 1 = fraud)
- **Tantangan**: Dataset sangat tidak seimbang (~3.5% fraud)

### Model yang Digunakan
| Komponen | Detail |
|----------|--------|
| Arsitektur | MLP 3-Layer (256 → 128 → 64 → 1) |
| Aktivasi | ReLU (hidden), Sigmoid (output) |
| Regularisasi | Batch Normalization + Dropout |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Class Imbalance | SMOTE (sampling_strategy=0.3) |

### Metrik Evaluasi
| Metrik | Keterangan |
|--------|-----------|
| **AUC-ROC** | Area di bawah kurva ROC — metrik utama untuk imbalanced data |
| **F1-Score** | Harmonic mean antara Precision dan Recall |
| **Precision** | Dari semua prediksi fraud, berapa yang benar fraud |
| **Recall** | Dari semua fraud sebenarnya, berapa yang terdeteksi |
| **Confusion Matrix** | Visualisasi TP, TN, FP, FN |

### Hyperparameter yang Di-tune (Optuna)
- `units_1`, `units_2`, `units_3`: Ukuran layer
- `dropout_1`, `dropout_2`: Dropout rate
- `learning_rate`: Learning rate Adam
- `batch_size`: Ukuran batch

---

## 📊 Task 2: Song Year Regression

### Dataset
- **File**: `midterm-regresi-dataset.csv`
- **Sumber**: [Google Drive](https://drive.google.com/file/d/1f8eaAZY-7YgFxLcrL3OkvSRa3onNNLb9/view)
- **Deskripsi**: Kolom pertama = tahun rilis lagu (target), kolom sisanya = fitur audio numerik
- **Fitur**: Representasi timbre dan karakteristik audio (feature_1, feature_2, ...)

### Model yang Digunakan
| Komponen | Detail |
|----------|--------|
| Arsitektur | MLP Regresi (jumlah layer dinamis) + LightGBM (Model Pembanding) |
| Aktivasi | ReLU (hidden), Linear (output) |
| Regularisasi | Batch Normalization + Dropout |
| Loss Function | Huber Loss (robust terhadap outlier) |
| Optimizer | Adam |
| Feature Eng. | Polynomial features (kuadrat) untuk top-5 fitur berkorelasi |

### Metrik Evaluasi
| Metrik | Keterangan |
|--------|-----------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root MSE — interpretable dalam satuan tahun |
| **MAE** | Mean Absolute Error — rata-rata selisih absolut |
| **R²** | Koefisien determinasi (1.0 = sempurna) |

### Interpretasi Model: LIME
- **LIME (Local Interpretable Model-agnostic Explanations)** digunakan untuk menjelaskan prediksi individual
- Menampilkan kontribusi tiap fitur terhadap prediksi model
- Dilakukan untuk 3 sampel individual + analisis global (100 sampel)

### Hyperparameter yang Di-tune (Optuna)
- `n_layers`: Jumlah hidden layer (2–5)
- `units`: Ukuran tiap layer
- `dropout_rate`: Dropout rate
- `learning_rate`: Learning rate Adam
- `batch_size`: Ukuran batch

---

## 🚀 Cara Menjalankan Notebook

### 1. Persiapan Environment

```bash
# Install semua dependensi
pip install tensorflow optuna mlflow imbalanced-learn lime scikit-learn pandas numpy matplotlib seaborn lightgbm
```

### 2. Persiapan Dataset

Letakkan file dataset di folder yang sesuai:
- `Task1_FraudDetection/train_transaction.csv`
- `Task2_SongYearRegression/midterm-regresi-dataset.csv`

### 3. Jalankan Notebook

Buka Jupyter Notebook atau JupyterLab:

```bash
jupyter notebook
```

Navigasi ke folder yang sesuai dan jalankan notebook secara berurutan (Run All Cells).

### 4. Lihat MLFlow Dashboard

Setelah training selesai, buka terminal di folder task yang diinginkan dan jalankan:

```bash
mlflow ui
```

Buka browser dan akses: **http://localhost:5000**

---

## 📦 Library yang Digunakan

| Library | Versi | Kegunaan |
|---------|-------|----------|
| TensorFlow/Keras | ≥2.10 | Framework deep learning |
| Optuna | ≥3.0 | Hyperparameter tuning |
| MLFlow | ≥2.0 | Experiment tracking |
| imbalanced-learn | ≥0.10 | SMOTE (Task 1) |
| LIME | ≥0.2 | Model interpretation (Task 2) |
| LightGBM | ≥3.3 | Model pembanding (Task 2) |
| scikit-learn | ≥1.0 | Preprocessing & metrics |
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Komputasi numerik |
| matplotlib | ≥3.6 | Visualisasi |
| seaborn | ≥0.12 | Visualisasi statistik |

---

## 📝 Navigasi Notebook

### Task 1 — `Task1_FraudDetection/Task1_FraudDetection.ipynb`

| Section | Deskripsi |
|---------|-----------|
| Bagian 0 | Instalasi library |
| Bagian 1 | Import library |
| Bagian 2 | Load dataset |
| Bagian 3 | Eksplorasi Data (EDA) — distribusi kelas, missing values, visualisasi |
| Bagian 4 | Preprocessing — encoding, imputasi, SMOTE, normalisasi |
| Bagian 5 | Arsitektur model MLP |
| Bagian 6 | Hyperparameter tuning dengan Optuna (15 trials) |
| Bagian 7 | Training model final + MLFlow tracking |
| Bagian 8 | Evaluasi — confusion matrix, ROC curve, threshold analysis |
| Bagian 9 | MLFlow dashboard guide |
| Bagian 10 | Kesimpulan |

### Task 2 — `Task2_SongYearRegression/Task2_SongYearRegression.ipynb`

| Section | Deskripsi |
|---------|-----------|
| Bagian 0 | Instalasi library |
| Bagian 1 | Import library |
| Bagian 2 | Load dataset |
| Bagian 3 | Eksplorasi Data (EDA) — distribusi tahun, korelasi fitur |
| Bagian 4 | Preprocessing — imputasi, IQR clipping, feature engineering, normalisasi |
| Bagian 5 | Arsitektur model MLP regresi |
| Bagian 6 | Hyperparameter tuning dengan Optuna (15 trials) |
| Bagian 7 | Training model final + MLFlow tracking |
| Bagian 8 | Training model pembanding LightGBM |
| Bagian 9 | Evaluasi — actual vs predicted, residual analysis |
| Bagian 10 | Interpretasi LIME (lokal + global) |
| Bagian 11 | MLFlow dashboard guide |
| Bagian 12 | Kesimpulan |

---

*Generated with ❤️ for UAS Machine Learning*
