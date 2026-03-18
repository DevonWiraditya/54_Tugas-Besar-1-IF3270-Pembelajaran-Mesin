# 54_Tugas-Besar-1-IF3270-Pembelajaran-Mesin

## Deskripsi Singkat
Repository ini berisi tugas besar 1 IF3270 Pembelajaran Mesin.
Isi utamanya meliputi implementasi komponen FFNN dari nol (autodiff, fungsi aktivasi, dan model), serta notebook untuk eksperimen dan evaluasi model pada dataset.

## Setup
Pastikan Python 3.10+ sudah terpasang, lalu install dependency utama:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Cara Menjalankan
Buka notebook langsung di VS Code, lalu jalankan sel:
- `notebooks/notebook.ipynb`

## Struktur Singkat
- `data/`: dataset
- `src/`: implementasi kode utama (`autodiff.py`, `functions.py`, `nn.py`)
- `notebooks/`: eksperimen, training, dan evaluasi
- `doc/`: laporan

## Pembagian Tugas
| Nama | NIM | Tugas |
|---|---|---|
| Muhammad Adam M | 18223015 | Membuat `autodiff.py`, membuat `functions.py`, membuat laporan bagian Penjelasan Implementasi, membuat laporan bagian Deskripsi Persoalan |
| Muhammad Azzam R | 18223025 | Melakukan fix pada `notebook.ipynb`, membuat `nn.py`, membuat laporan |
| Devon Wiraditya T | 18223039 | Membuat `notebook.ipynb`, melakukan fix pada `autodiff.py`, mengimplementasi optimizer Adam |