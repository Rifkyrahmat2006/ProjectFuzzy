# LAPORAN PROYEK: SISTEM PAKAR KUALITAS TIDUR (SLEEPZY)

## 1. Identifikasi Proyek

- **Nama Sistem:** SLEEPZY (Sistem Pakar Kualitas Tidur Mahasiswa)
- **Metode:** Logika Fuzzy Mamdani
- **Platform:** Web-based (Streamlit)
- **Tujuan:** Mengevaluasi kualitas tidur mahasiswa berdasarkan parameter gaya hidup menggunakan pendekatan sistem pakar.

---

## 2. Metodologi

Sistem ini mengimplementasikan Logika Fuzzy Mamdani dengan empat tahapan utama:

1.  **Fuzzifikasi:** Mengubah input _crisp_ (nilai riil) menjadi nilai linguistik menggunakan fungsi keanggotaan.
2.  **Pembentukan Rule Base:** Menentukan hubungan logika antara variabel input dan output.
3.  **Inferensi:** Menggunakan operator MIN untuk implikasi dan MAX untuk agregasi antar aturan.
4.  **Defuzzifikasi:** Mengubah hasil agregasi fuzzy kembali menjadi nilai _crisp_ menggunakan metode **Centroid**.

---

## 3. Spesifikasi Variabel

### 3.1 Variabel Input (Antecedents)

Sistem menggunakan tiga variabel input yang diambil dari pola hidup harian:

| Variabel            | Satuan  | Rentang Semesta | Himpunan Fuzzy          |
| :------------------ | :------ | :-------------- | :---------------------- |
| **Durasi Tidur**    | Jam     | 4 - 9           | Pendek, Sedang, Panjang |
| **Aktivitas Fisik** | Skor    | 0 - 120         | Rendah, Sedang, Tinggi  |
| **Konsumsi Kafein** | Cangkir | 0 - 5           | Rendah, Sedang, Tinggi  |

### 3.2 Variabel Output (Consequent)

Output akhir merupakan skor representasi kualitas tidur:

| Variabel           | Rentang | Himpunan Fuzzy                  |
| :----------------- | :------ | :------------------------------ |
| **Kualitas Tidur** | 1 - 10  | Buruk, Cukup, Baik, Sangat Baik |

---

## 4. Basis Aturan (Rule Base)

Sistem ini menggunakan **24 aturan logika IF-THEN** yang didefinisikan secara eksplisit untuk mencakup berbagai skenario pola tidur:

| No  | Durasi Tidur | Aktivitas Fisik | Konsumsi Kafein | Kualitas Tidur  |
| :-- | :----------- | :-------------- | :-------------- | :-------------- |
| 1   | Pendek       | Rendah          | Tinggi          | **Buruk**       |
| 2   | Pendek       | Rendah          | Sedang          | **Buruk**       |
| 3   | Pendek       | Rendah          | Rendah          | **Buruk**       |
| 4   | Pendek       | Sedang          | Tinggi          | **Buruk**       |
| 5   | Pendek       | Sedang          | Sedang          | **Cukup**       |
| 6   | Pendek       | Sedang          | Rendah          | **Cukup**       |
| 7   | Pendek       | Tinggi          | Tinggi          | **Cukup**       |
| 8   | Pendek       | Tinggi          | Sedang          | **Cukup**       |
| 9   | Pendek       | Tinggi          | Rendah          | **Cukup**       |
| 10  | Sedang       | Rendah          | Tinggi          | **Buruk**       |
| 11  | Sedang       | Rendah          | Sedang          | **Cukup**       |
| 12  | Sedang       | Rendah          | Rendah          | **Cukup**       |
| 13  | Sedang       | Sedang          | Tinggi          | **Cukup**       |
| 14  | Sedang       | Sedang          | Sedang          | **Baik**        |
| 15  | Sedang       | Sedang          | Rendah          | **Baik**        |
| 16  | Sedang       | Tinggi          | Tinggi          | **Cukup**       |
| 17  | Sedang       | Tinggi          | Sedang          | **Baik**        |
| 18  | Sedang       | Tinggi          | Rendah          | **Sangat Baik** |
| 19  | Panjang      | Rendah          | Tinggi          | **Cukup**       |
| 20  | Panjang      | Rendah          | Sedang          | **Cukup**       |
| 21  | Panjang      | Rendah          | Rendah          | **Baik**        |
| 22  | Panjang      | Sedang          | Tinggi          | **Cukup**       |
| 23  | Panjang      | Sedang          | Sedang          | **Baik**        |
| 24  | Panjang      | Sedang          | Rendah          | **Sangat Baik** |

---

## 5. Laporan Pengujian (Testing Report)

### 5.1 Metodologi Pengujian

Pengujian dilakukan menggunakan 10 baris data pertama dari dataset `student_sleep_patterns.csv`. Sistem mengeksekusi mesin fuzzy untuk setiap baris data dan membandingkan hasil defuzzifikasi dengan ekspektasi logika pakar.

### 5.2 Rincian Logika Pengujian

Setiap pengujian mengikuti alur logika internal berikut:

1.  **Ekstraksi Data:** Nilai _crisp_ untuk `Sleep_Duration`, `Physical_Activity`, dan `Caffeine_Intake` dibaca dari dataset.
2.  **Kalkulasi Derajat Keanggotaan:**
    - Sistem menghitung sejauh mana nilai input masuk ke dalam setiap himpunan fuzzy (misal: seberapa "Sedang" durasi tidurnya).
3.  **Evaluasi Aturan (Fuzzy Inference):**
    - Setiap dari 24 aturan diperiksa kekuatannya menggunakan operator **AND (MIN)**. Jika salah satu input memiliki derajat 0 pada himpunan yang disyaratkan aturan, maka aturan tersebut tidak aktif.
4.  **Agregasi:**
    - Semua aturan yang aktif digabungkan menggunakan operator **OR (MAX)** untuk membentuk satu area fuzzy baru pada variabel output.
5.  **Defuzzifikasi Centroid:**
    - Titik keseimbangan (pusat gravitasi) dari area agregasi dihitung. Titik inilah yang menjadi skor akhir Kualitas Tidur (1-10).

### 5.3 Hasil Uji Sampel (Batch Test)

Berdasarkan visualisasi pada aplikasi web, berikut adalah beberapa sampel hasil uji:

| Sampel     | Durasi (Jam) | Aktivitas | Kafein | Skor Kualitas (Fuzzy) | Analisis Logika                                                          |
| :--------- | :----------- | :-------- | :----- | :-------------------- | :----------------------------------------------------------------------- |
| **Data 1** | 7.7          | 37        | 2      | **6.73**              | Durasi panjang & kafein sedang mendukung kualitas "Baik".                |
| **Data 2** | 6.3          | 74        | 5      | **3.74**              | Kafein tinggi & durasi sedang-pendek menarik skor ke arah "Cukup/Buruk". |
| **Data 4** | 4.1          | 54        | 2      | **3.01**              | Durasi sangat pendek mendominasi hasil ke kualitas "Buruk".              |

---

## 6. Implementasi Sistem

### 5.1 Arsitektur Kode

Sistem dibangun secara modular untuk memastikan skalabilitas:

- `fuzzy_mamdani.py`: Modul inti yang berisi logika komputasi fuzzy, definisi keanggotaan, dan generator grafik menggunakan `matplotlib`.
- `app.py`: Antarmuka web menggunakan `Streamlit` yang mengintegrasikan pengolahan data CSV dan visualisasi batch.

### 5.2 Visualisasi Antarmuka

Aplikasi web menampilkan batch processing untuk 10 data pertama dari dataset. Setiap data divisualisasikan dalam grid 2x2 yang mencakup:

1.  Posisi input pada kurva **Durasi Tidur**.
2.  Posisi input pada kurva **Aktivitas Fisik**.
3.  Posisi input pada kurva **Kafein**.
4.  Area hasil **Inferensi** dan titik **Centroid** (skor akhir).

---

## 6. Deployment

Sistem dikonfigurasi untuk dapat dideploy pada **Streamlit Cloud** dengan spesifikasi:

- **Runtime:** Python 3.11 (via `runtime.txt`).
- **Dependensi:** Tercatat pada `requirements.txt` (termasuk `scikit-fuzzy`, `networkx`, dan `scipy`).

---

## 7. Kesimpulan

Sistem SLEEPZY membuktikan bahwa logika fuzzy efektif dalam menangani ambiguitas data kualitatif seperti "Kualitas Tidur". Dengan visualisasi yang transparan, sistem ini tidak hanya memberikan angka akhir tetapi juga memberikan penjelasan visual mengenai bagaimana angka tersebut dihasilkan dari interaksi antar aturan pakar.
