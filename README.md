# Dokumentasi Proyek Data Science Binar Academy

## Deskripsi
Proyek ini merupakan bagian dari bootcamp Data Science di Binar Academy, yang bertujuan untuk
mengembangkan model machine learning untuk melakukan analisis sentimen terhadap data Twitter —
menggunakan dataset yang telah diklasifikasikan menjadi label **positive**, **negative**, dan
**neutral**.

Dokumentasi lengkap (EDA, model, dan tampilan API) disajikan sebagai dashboard Streamlit:

https://bathis-ds-dasboard.streamlit.app/

## Struktur Proyek
- **Dasboard.py** — dashboard dokumentasi (Streamlit).
- **Data Training/** — dataset training `train_preprocess.tsv` (11.000 tweet beserta label sentimen).
- **DataKlasifikasi/** — file pendukung cleansing data: `new_kamusalay.csv` (kamus kata alay),
  `abusive.csv` (daftar kata kasar), dan `data.csv` (dataset awal berlabel HS).
- **Templates/** — gambar dokumentasi (grafik, word cloud, evaluasi model, tampilan API) beserta
  file HTML/CSS/JS antarmuka API.
- **.streamlit/config.toml** — tema tampilan dashboard.

## Library yang Digunakan
Terinstal otomatis melalui `requirements.txt`:
- `streamlit`
- `pandas`
- `Pillow`

## Cara Menjalankan Dashboard
```bash
pip install -r requirements.txt
streamlit run Dasboard.py
```

Atau buka link deployment di atas.
