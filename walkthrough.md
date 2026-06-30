# Walkthrough - Ship Structural Reliability Streamlit App

Dokumentasi ini dibuat sebagai referensi untuk pengembangan dan analisis keandalan struktur kapal menggunakan aplikasi Streamlit yang telah dibuat di folder `app/`.

---

## 🏗️ Arsitektur Aplikasi

Mengikuti panduan `streamlit-guidelines.md`, aplikasi dipisahkan menjadi dua lapisan utama untuk memastikan modularitas dan kemudahan pemeliharaan:

```
User Interface (Streamlit)
       │ (app.py)
       ▼
Scientific Calculation Engine (Pure Python)
       │
       ├── src/loader.py       <-- Memuat dan memproses CSV (Turkstra Rule)
       ├── src/calculations.py <-- Menghitung keandalan numerik (Level 3)
       └── src/plotting.py     <-- Visualisasi Matplotlib untuk Streamlit
```

---

## 🔢 Rumus Penting & Logika Sains

### 1. Turkstra's Rule untuk Penggabungan Beban Dinamis (3 Komponen)
Beban momen gelombang dinamis dihitung dengan menggabungkan momen vertikal, horizontal, dan torsional dengan faktor korelasi dari deret waktu:
$$WBM_{\text{dynamic}}(t) = |f_1(t)| + K_2 |f_2(t)| + K_3 |f_3(t)|$$
di mana:
- $f_1(t)$ adalah beban dominan:
  * Arah gelombang $180^\circ$ (Head Wave): Dominan = Momen Vertikal.
  * Arah gelombang $90^\circ$ (Beam Wave): Dominan = Momen Horizontal.
- $K_2 = |\rho_{12}|$ dan $K_3 = |\rho_{13}|$ adalah nilai absolut koefisien korelasi Pearson antara beban dominan dengan beban sekunder/tersier.

### 2. Konversi Tegangan Momen (MPa)
Momen lentur total ($M$) dikonversi ke tegangan ($\sigma_L$) dalam satuan Megapascal (MPa) menggunakan Modulus Penampang ($W$) yang diinput langsung oleh pengguna:
$$\sigma_L \text{ (MPa)} = \frac{M \text{ (N.m)}}{W \text{ (m}^3\text{)}} \times 10^{-6}$$

### 3. Keandalan Level 3 (Integral Konvolusi)
Peluang kegagalan ($P_f$) dihitung dengan mengintegrasikan fungsi kumulatif kekuatan ($F_S$) terhadap fungsi densitas beban ($f_L$):
$$P_f = \int_0^\infty F_S(x) f_L(x) dx$$
- Aplikasi ini mendukung distribusi **Normal**, **Lognormal**, dan **Weibull** untuk Kekuatan ($S$).
- Untuk Beban ($L$), didukung **Normal**, **Rayleigh**, dan **Weibull**.
- Penyelesaian numerik dilakukan menggunakan `scipy.integrate.quad`.
- Indeks Keamanan Ekuivalen ($\beta_{L3}$) dihitung dari:
  $$\beta_{L3} = -\Phi^{-1}(P_f)$$

---

## 🌊 Informasi Pemetaan Data Contoh (Contoh Berkas)

Sesuai hasil analisis data gelombang aktual:
- File `wbm/Hs-1.73_D-180.csv` memiliki momen paling besar, sehingga dipetakan secara fisik sebagai ketinggian gelombang **$H_s = 2.58\text{ m}$**.
- File `wbm/Hs-2.58_D-180.csv` memiliki momen paling kecil, sehingga dipetakan secara fisik sebagai ketinggian gelombang **$H_s = 1.73\text{ m}$**.

Aplikasi Streamlit secara otomatis menerapkan pemetaan fisik yang konsisten ini saat memuat contoh data agar hasil grafik perbandingan menunjukkan tren yang tepat (semakin besar $H_s$, nilai keandalan/safety index semakin kecil/kritis).

---

## 🛠️ Cara Menjalankan Aplikasi
1. Jalankan shell dan aktifkan environment:
   ```bash
   source .venv/bin/activate
   ```
2. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Klik tombol **"Load Default Notebook Data"** di sidebar untuk langsung memuat konfigurasi analisis standar 12-kasus yang sesuai dengan Jupyter Notebook.
