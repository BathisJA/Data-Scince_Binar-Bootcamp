"""Dokumentasi proyek analisis sentimen Twitter — Binar Academy Data Science Bootcamp.

Jalankan dengan: streamlit run Dasboard.py
"""

from pathlib import Path
import re

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

# ------------------------------------------------------- pembersihan data ---

URL_RE = re.compile(r"(?:https?://|www\.)\S+")
NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")
TOKEN_RE = re.compile(r"\b(?:rt|user|url)\b")
XNN_RE = re.compile(r"x[a-zA-Z0-9]{2}")
PLUS_N_RE = re.compile(r"\+n")
MULTI_SPACE_RE = re.compile(r"  +")


def clean_character(text):
    s = str(text).lower()
    s = URL_RE.sub(" ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = TOKEN_RE.sub(" ", s)
    # buang karakter non-ascii yang ter-escape (emoji, dsb.)
    s = s.encode("unicode_escape").decode("utf-8")
    s = XNN_RE.sub("", s)
    s = s.encode("utf-8").decode("unicode_escape", errors="ignore")
    s = s.replace(":", " ").replace(";", " ")
    s = PLUS_N_RE.sub(" ", s)
    s = s.replace("\n", " ").replace("+", " ")
    return MULTI_SPACE_RE.sub(" ", s).strip()


def build_censor_regex(words):
    """Satu regex utuh dengan word boundary — hanya mencocokkan kata utuh.

    Kata terpanjang dicocokkan lebih dulu agar tidak saling menimpa.
    """
    escaped = sorted((re.escape(str(w).lower()) for w in words), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def mask_word(match):
    w = match.group()
    # kata pendek (<=2 huruf) cukup huruf pertama + satu tanda bintang
    return w[0] + "*" * (len(w) - 2) + w[-1] if len(w) > 2 else w[0] + "*"


@st.cache_data(show_spinner=False)
def load_prepare():
    """Muat dataset dan jalankan pipeline pembersihan (hasil di-cache Streamlit)."""
    df_train = pd.read_csv(
        ROOT / "Data Training" / "train_preprocess.tsv", sep="\t", names=["text", "label"]
    )
    df_slang = pd.read_csv(
        ROOT / "DataKlasifikasi" / "new_kamusalay.csv", encoding="latin-1", names=["alay", "kbbi"]
    ).dropna()
    df_abusive = pd.read_csv(ROOT / "DataKlasifikasi" / "abusive.csv", encoding="latin-1")

    alay_map = dict(zip(df_slang["alay"].astype(str), df_slang["kbbi"].astype(str)))

    def alay_cleanse(text):
        return " ".join(alay_map.get(w, w) for w in text.split(" "))

    censor_re = build_censor_regex(df_abusive["ABUSIVE"])

    def clean_pipeline(text):
        return censor_re.sub(mask_word, alay_cleanse(clean_character(text)))

    df_train["cleaned_text"] = df_train["text"].astype(str).map(clean_pipeline)
    df_train["total char"] = df_train["cleaned_text"].str.len()
    df_train["total word"] = df_train["cleaned_text"].str.split().str.len()

    stats = {
        "rows": len(df_train),
        "n_slang": len(df_slang),
        "n_abusive": int(df_abusive["ABUSIVE"].size),
        "duplicates": int(df_train.duplicated().sum()),
        "nulls": df_train[["text", "label"]].isnull().sum(),
        "mean_char": float(df_train["total char"].mean()),
        "mean_word": float(df_train["total word"].mean()),
        "label_counts": df_train["label"].value_counts(),
    }
    return df_train, stats


# ------------------------------------------------------------- util format ---

def fmt_id(n):
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(p):
    return f"{p:.1f}".replace(".", ",") + "%"


def fmt_1d(x):
    return f"{x:.1f}".replace(".", ",")


# ------------------------------------------------------------------- tema ---

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900&family=Archivo:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root{
  --paper:#f5f0e6;
  --paper-2:#ece5d3;
  --card:#fffcf5;
  --ink:#1c1914;
  --ink-2:#6f6555;
  --line:#d9cfba;
  --accent:#1467b6;
  --pos:#1e7a52;
  --neg:#c2402f;
  --neu:#8a7f6a;
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  --sans:'Archivo','Helvetica Neue',Arial,sans-serif;
  --mono:'IBM Plex Mono','Courier New',monospace;
}
::selection{background:var(--accent);color:#fff}
.stApp{background:var(--paper);color:var(--ink);font-family:var(--sans)}
.stApp::before{content:"";position:fixed;inset:0;z-index:999;pointer-events:none;opacity:.55;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.11 0 0 0 0 0.10 0 0 0 0 0.08 0 0 0 0.05 0'/></filter><rect width='160' height='160' filter='url(%23n)'/></svg>")}
[data-testid="stHeader"]{background:transparent}
#MainMenu{visibility:hidden}
footer{visibility:hidden}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:var(--line);border:3px solid var(--paper);border-radius:8px}
.block-container{max-width:1120px;padding:2.4rem 2.4rem 4.5rem}
[data-testid="stMarkdownContainer"] p{line-height:1.7}

[data-testid="stSidebar"]{background:var(--paper-2);border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{max-width:100%;padding:1.8rem 1.3rem 3rem}

[data-testid="stCaptionContainer"]{font-family:var(--mono);font-size:.76rem;color:var(--ink-2);letter-spacing:.02em}
[data-testid="stImage"] img{width:100%;background:var(--card);border:1px solid var(--ink);box-shadow:6px 6px 0 rgba(28,25,20,.12)}
[data-testid="stDataFrame"]{border:1px solid var(--line)}

.hero{border-bottom:2px solid var(--ink);padding-bottom:2rem;margin-bottom:2.4rem}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.24em;text-transform:uppercase;color:var(--accent);margin-bottom:1.2rem}
.stApp h1.hero-title{font-family:var(--serif);font-size:clamp(2.5rem,4.6vw,3.9rem);line-height:1.05;font-weight:600;letter-spacing:-.015em;margin:0 0 1.3rem}
.hero-title em{font-style:italic;font-weight:500}
.standfirst{font-size:1.07rem;line-height:1.7;max-width:64ch;margin:0}
.standfirst::first-letter{font-family:var(--serif);font-size:3.2em;float:left;line-height:.82;padding:.04em .12em 0 0;font-weight:600}
.metabar{display:flex;flex-wrap:wrap;gap:.55rem 2.4rem;margin-top:1.7rem;padding-top:1.05rem;border-top:1px solid var(--line)}
.metabar span{font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-2)}
.metabar b{color:var(--ink);font-weight:600}

.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--ink);background:var(--card);box-shadow:6px 6px 0 rgba(28,25,20,.12);margin:0 0 1rem}
.stat{padding:1.15rem 1.25rem 1.05rem;border-left:1px solid var(--line)}
.stat:first-child{border-left:none}
.stat-n{font-family:var(--serif);font-size:2.05rem;font-weight:600;line-height:1}
.stat-l{font-family:var(--mono);font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);margin-top:.5rem;line-height:1.6}
@media(max-width:760px){.stats{grid-template-columns:repeat(2,1fr)}.stat:nth-child(n+3){border-top:1px solid var(--line)}.stat:nth-child(3){border-left:none}}

.sec{display:flex;align-items:baseline;gap:1.2rem;margin:3.4rem 0 1.5rem;padding-top:1.3rem;border-top:1px solid var(--ink)}
.sec-num{font-family:var(--serif);font-style:italic;font-size:1.45rem;color:var(--accent)}
.stApp h2.sec-title{font-family:var(--serif);font-size:2.05rem;font-weight:600;letter-spacing:-.01em;margin:0;line-height:1.1}

.prose{max-width:70ch;font-size:.96rem;line-height:1.75;margin:.2rem 0 .9rem}
.stApp .subhead{font-family:var(--serif);font-size:1.42rem;font-weight:600;margin:1.9rem 0 .45rem;line-height:1.2}

.steps{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);background:var(--card);margin:.4rem 0 1.6rem}
.step{padding:1.05rem 1.2rem;border-left:1px solid var(--line)}
.step:first-child{border-left:none}
.step-n{font-family:var(--mono);font-size:.68rem;letter-spacing:.18em;color:var(--accent)}
.step b{display:block;font-family:var(--serif);font-size:1.06rem;font-weight:600;margin:.4rem 0 .32rem}
.step p{font-size:.84rem;line-height:1.6;color:var(--ink-2);margin:0}
@media(max-width:760px){.steps{grid-template-columns:1fr}.step{border-left:none;border-top:1px solid var(--line)}.step:first-child{border-top:none}}

.labar-track{display:flex;height:2.5rem;border:1px solid var(--ink);background:var(--card);overflow:hidden}
.labar-track div{height:100%}
.labar-legend{display:flex;flex-wrap:wrap;gap:.45rem 1.7rem;margin-top:.75rem;font-family:var(--mono);font-size:.75rem;color:var(--ink)}
.dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;margin-right:.5rem}

.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.9rem 0 .3rem}
.chips span{font-family:var(--mono);font-size:.73rem;border:1px solid var(--ink);background:var(--card);padding:.3rem .62rem}

.foot{display:flex;justify-content:space-between;flex-wrap:wrap;gap:.4rem 1.5rem;margin-top:4rem;padding-top:1.05rem;border-top:1px solid var(--ink);font-family:var(--mono);font-size:.7rem;letter-spacing:.07em;color:var(--ink-2);text-transform:uppercase}

.sb-mark{font-family:var(--mono);font-size:.66rem;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);margin-bottom:.9rem}
.stApp .sb-title{font-family:var(--serif);font-size:1.45rem;font-weight:600;margin:0 0 .5rem}
.sb-note{font-size:.85rem;line-height:1.65;color:var(--ink-2);margin:0}
.sb-overline{font-family:var(--mono);font-size:.66rem;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-2);margin:1.5rem 0 .55rem;padding-top:1rem;border-top:1px solid var(--line)}
.sb-line{font-family:var(--mono);font-size:.77rem;margin:.3rem 0}
.sb-line b{font-weight:600}
"""


def md(html):
    st.markdown(html, unsafe_allow_html=True)


def section(num, title):
    md(f'<div class="sec"><span class="sec-num">{num}</span><h2 class="sec-title">{title}</h2></div>')


def subhead(text):
    md(f'<div class="subhead">{text}</div>')


def figure(filename, num, caption):
    st.image(str(ROOT / "Templates" / filename))
    st.caption(f"Fig. {num:02d} — {caption}")


def chips(items):
    md('<div class="chips">' + "".join(f"<span>{i}</span>" for i in items) + "</div>")


# ------------------------------------------------------------------- app ---

st.set_page_config(
    page_title="Analisis Sentimen Twitter — Binar Academy",
    page_icon=":bird:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.spinner("Menyiapkan data…"):
    df_train, stats = load_prepare()

LABEL_COLORS = {"positive": "var(--pos)", "negative": "var(--neg)", "neutral": "var(--neu)"}
LABELS = ["positive", "negative", "neutral"]
counts = {lab: int(stats["label_counts"].get(lab, 0)) for lab in LABELS}
total = stats["rows"]
pcts = {lab: counts[lab] / total * 100 for lab in LABELS}

# ------------------------------------------------------------- sidebar ---

with st.sidebar:
    md('<div class="sb-mark">Binar Academy · DS Bootcamp</div>')
    md('<div class="sb-title">Catatan Dataset</div>')
    md('<p class="sb-note">Detail mengenai dataset yang digunakan untuk melakukan '
       'training pada Machine Learning.</p>')
    md('<div class="sb-overline">Sumber Data</div>')
    md('<div class="sb-line"><b>train_preprocess.tsv</b></div>')
    md(f'<div class="sb-line">{fmt_id(total)} baris · 2 kolom</div>')
    md('<div class="sb-overline">Null Values</div>')
    for col, n in stats["nulls"].items():
        md(f'<div class="sb-line">{col} — <b>{fmt_id(n)}</b></div>')
    md('<div class="sb-overline">Duplicate Values</div>')
    md(f'<div class="sb-line">Data duplikat: <b>{fmt_id(stats["duplicates"])}</b></div>')
    md('<div class="sb-overline">Rata-rata Jumlah Karakter &amp; Kata</div>')
    md(f'<div class="sb-line">total char — <b>{fmt_1d(stats["mean_char"])}</b></div>')
    md(f'<div class="sb-line">total word — <b>{fmt_1d(stats["mean_word"])}</b></div>')
    md('<div class="sb-overline">Distribusi Label</div>')
    for lab in LABELS:
        md(f'<div class="sb-line"><i class="dot" style="background:{LABEL_COLORS[lab]}"></i>'
           f'{lab} — <b>{fmt_id(counts[lab])}</b></div>')

# ----------------------------------------------------------------- hero ---

md(f"""
<div class="hero">
  <div class="eyebrow">Binar Academy · Data Science Bootcamp — Dokumentasi Proyek</div>
  <h1 class="hero-title">Analisis Sentimen <em>Twitter</em> Indonesia</h1>
  <p class="standfirst">Melakukan analisis sentimen terhadap data Twitter menggunakan metode
  <b>Neural Network</b> dan <b>Long Short-Term Memory (LSTM)</b> melalui API. Di bawah ini adalah
  hasil dari analisis data yang telah dilakukan, beserta contoh API yang sudah dibuat.</p>
  <div class="metabar">
    <span><b>{fmt_id(total)}</b>&nbsp;tweet data train</span>
    <span><b>2</b>&nbsp;arsitektur model</span>
    <span><b>Flask</b>&nbsp;+ Swagger API</span>
  </div>
</div>
""")

md(f"""
<div class="stats">
  <div class="stat"><div class="stat-n">{fmt_id(total)}</div><div class="stat-l">tweet pada data train</div></div>
  <div class="stat"><div class="stat-n">3</div><div class="stat-l">label sentimen — positive · negative · neutral</div></div>
  <div class="stat"><div class="stat-n">{fmt_id(stats["n_slang"])}</div><div class="stat-l">kosakata alay dinormalisasi</div></div>
  <div class="stat"><div class="stat-n">{fmt_id(stats["n_abusive"])}</div><div class="stat-l">kata abusive disensor</div></div>
</div>
""")

# ------------------------------------------- 01 · exploratory data analysis ---

section("01", "Exploratory Data Analysis")
md('<p class="prose">Ini adalah dokumentasi dari proyek Bootcamp Data Science dari Binar Academy. '
   '<b>Tujuan proyek:</b> melakukan analisis sentimen terhadap data Twitter menggunakan metode '
   'Neural Network dan Long Short-Term Memory melalui API. Di bawah ini adalah hasil dari analisis '
   'data yang telah dilakukan beserta contoh API yang sudah dibuat.</p>')

md("""
<div class="steps">
  <div class="step"><span class="step-n">TAHAP 01</span><b>Pembersihan teks</b>
  <p>Menghapus URL, emoji, dan karakter non-alfanumerik, lalu menyaring token rt, user, dan url dari setiap tweet.</p></div>
  <div class="step"><span class="step-n">TAHAP 02</span><b>Normalisasi kata alay</b>
  <p>Kata slang dipetakan ke kata baku menggunakan kamus alay pada new_kamusalay.csv.</p></div>
  <div class="step"><span class="step-n">TAHAP 03</span><b>Sensor kata abusive</b>
  <p>Kata kasar dari abusive.csv disensor dengan pencocokan kata utuh (word boundary), tanpa menyentuh kata lain.</p></div>
</div>
""")

subhead("Data yang Sudah Dibersihkan")
md('<p class="prose">Berikut ini adalah contoh data yang sudah melalui proses cleansing — mengganti '
   'kata slang dengan kata baku dan menyensor kata kasar — beserta label sentimen dari data tersebut. '
   'Data ini nantinya digunakan untuk melakukan training Machine Learning.</p>')
st.dataframe(
    df_train[["cleaned_text", "label"]].head(21),
    hide_index=True,
    width="stretch",
    column_config={
        "cleaned_text": st.column_config.TextColumn("Teks (setelah pembersihan)", width="large"),
        "label": st.column_config.TextColumn("Label", width="small"),
    },
)

subhead("Proporsi Label")
bar = "".join(
    f'<div style="width:{pcts[lab]:.2f}%;background:{LABEL_COLORS[lab]}"></div>' for lab in LABELS
)
legend = "".join(
    f'<span><i class="dot" style="background:{LABEL_COLORS[lab]}"></i>'
    f'{lab} — {fmt_pct(pcts[lab])} ({fmt_id(counts[lab])} tweet)</span>'
    for lab in LABELS
)
md(f'<div class="labar-track">{bar}</div><div class="labar-legend">{legend}</div>')
md('<p class="prose">Proporsi dihitung langsung dari data train. Secara garis besar, tweet berlabel '
   'positive mendominasi dataset, diikuti label negative dan neutral.</p>')
figure("piechart.png", 1, "Distribusi label pada data train — positive, negative, neutral.")

subhead("Persebaran Tweet")
md('<p class="prose">Secara keseluruhan, grafik menunjukkan bahwa tweet dalam dataset tersebut '
   'relatif pendek, dengan kebanyakan tweet memiliki kurang dari 400 karakter dan kurang dari 60 kata.</p>')
figure("barchart.png", 2, "Persebaran panjang tweet: jumlah karakter dan jumlah kata.")

c1, c2 = st.columns(2, gap="large")
with c1:
    subhead("Word Cloud Kata Kasar")
    figure("wordcloud_abusive.png", 3, "Kata-kata kasar yang sering digunakan dalam dataset.")
with c2:
    subhead("Word Cloud Kata Alay")
    figure("wordcloude_alay.png", 4, "Kata-kata alay yang sering digunakan dalam dataset.")

# -------------------------------------------------- 02 · model machine learning ---

section("02", "Model Machine Learning")
md('<p class="prose">Di bawah ini adalah visualisasi hasil evaluasi dari machine learning yang sudah '
   'dibuat, dilatih dengan K-Fold Cross Validation dan EarlyStopping.</p>')

subhead("Model LSTM")
figure("lstm.png", 5, "Kurva akurasi &amp; loss selama pelatihan model LSTM.")
md('<p class="prose">Layer yang kami gunakan dalam model:</p>')
chips(["KFold Cross Validation", "Sequential model", "Embedding Layer", "LSTM Layer",
       "Dense Layer", "Optimizer dan Loss Function", "EarlyStopping"])

subhead("Model NN")
figure("nn.png", 6, "Kurva akurasi &amp; loss selama pelatihan model Neural Network.")
md('<p class="prose">Layer yang kami gunakan dalam model:</p>')
chips(["KFold Cross Validation", "Sequential model", "Embedding Layer", "SimpleRNN Layer",
       "Dense Layer", "Optimizer dan Loss Function", "EarlyStopping"])

# ----------------------------------------------------------- 03 · tampilan api ---

section("03", "Tampilan API")
md('<p class="prose">Ini adalah contoh API yang dibuat menggunakan Flask dan Swagger. Tampilan API '
   'diubah menggunakan HTML, CSS, dan JavaScript agar lebih menarik dan mudah digunakan.</p>')

subhead("Tampilan Asli")
figure("swagger.png", 7, "Tampilan asli dokumentasi API berbasis Swagger.")

subhead("Landing Page")
figure("Landing Page.png", 8, "Landing page antarmuka API (HTML, CSS, JavaScript).")

subhead("Input Page")
figure("Input Page.png", 9, "Halaman input antarmuka API.")

subhead("Contoh Output")
figure("api.png", 10, "Contoh output analisis sentimen dari API.")

md('<div class="foot"><span>Binar Academy — Data Science Bootcamp</span>'
   '<span>Streamlit · pandas · Neural Network &amp; LSTM</span></div>')
