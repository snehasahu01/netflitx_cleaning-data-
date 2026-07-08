"""
Netflix Dataset — Data Cleaning Studio
A polished, interactive Streamlit app that showcases an end-to-end
data cleaning pipeline for a messy Netflix movies dataset.

Run locally:
    pip install streamlit pandas numpy plotly
    streamlit run app.py
"""

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Data Cleaning Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --bg:            #0B0C0F;
        --panel:         #15171C;
        --panel-line:    #262932;
        --red:           #D6303C;
        --amber:         #F5A623;
        --text:          #ECEDEF;
        --muted:         #8A8F9B;
        --good:          #3FBF7F;
    }

    .stApp{
        background:
            radial-gradient(circle at 15% 0%, rgba(214,48,60,0.10), transparent 45%),
            radial-gradient(circle at 85% 10%, rgba(245,166,35,0.06), transparent 40%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Headline / hero */
    .reel-eyebrow{
        font-family:'JetBrains Mono', monospace;
        letter-spacing:.25em;
        text-transform:uppercase;
        font-size:.72rem;
        color: var(--amber);
        margin-bottom:.4rem;
    }
    .reel-title{
        font-family:'Bebas Neue', sans-serif;
        font-size: 4.4rem;
        line-height: .95;
        letter-spacing: .02em;
        color: var(--text);
        margin: 0;
    }
    .reel-title span{ color: var(--red); }
    .reel-sub{
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 720px;
        margin-top:.6rem;
    }

    hr.reel-divider{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, var(--red), transparent 70%);
        margin: 1.6rem 0 1.8rem 0;
    }

    /* Cards */
    .card{
        background: var(--panel);
        border: 1px solid var(--panel-line);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
    }
    .metric-label{
        font-family:'JetBrains Mono', monospace;
        font-size:.68rem;
        letter-spacing:.12em;
        text-transform:uppercase;
        color: var(--muted);
    }
    .metric-value{
        font-family:'Bebas Neue', sans-serif;
        font-size: 2.5rem;
        color: var(--text);
        line-height:1.1;
    }
    .metric-value.red{ color: var(--red); }
    .metric-value.good{ color: var(--good); }
    .metric-value.amber{ color: var(--amber); }
    .metric-delta{ font-size:.78rem; color: var(--muted); margin-top:.2rem;}

    /* Step card in pipeline */
    .step-num{
        font-family:'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        color: var(--red);
        border: 1px solid var(--red);
        border-radius: 50%;
        width: 40px; height: 40px;
        display:flex; align-items:center; justify-content:center;
    }

    .badge{
        display:inline-block;
        font-family:'JetBrains Mono', monospace;
        font-size:.68rem;
        letter-spacing:.08em;
        text-transform:uppercase;
        padding: .18rem .55rem;
        border-radius: 999px;
        border: 1px solid var(--panel-line);
        color: var(--muted);
        margin-right:.4rem;
    }
    .badge.fixed{ color: var(--good); border-color: var(--good); }
    .badge.issue{ color: var(--red); border-color: var(--red); }

    section[data-testid="stSidebar"]{
        background: #0E0F13;
        border-right: 1px solid var(--panel-line);
    }

    .stTabs [data-baseweb="tab-list"]{ gap: 4px; }
    .stTabs [data-baseweb="tab"]{
        background: var(--panel);
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--panel-line);
        color: var(--muted);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"]{
        color: var(--text) !important;
        border-bottom: 2px solid var(--red) !important;
    }

    div[data-testid="stDataFrame"]{
        border: 1px solid var(--panel-line);
        border-radius: 8px;
        overflow:hidden;
    }

    .stButton>button, .stDownloadButton>button{
        background: var(--red);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: .5rem 1.1rem;
    }
    .stButton>button:hover, .stDownloadButton>button:hover{
        background: #b8232d;
        color: white;
    }

    footer.credit{
        color: var(--muted);
        font-size:.8rem;
        text-align:center;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--panel-line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECEDEF", family="Inter"),
        colorway=["#D6303C", "#F5A623", "#3FBF7F", "#8A8F9B", "#5B8DEF"],
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#262932", zerolinecolor="#262932"),
        yaxis=dict(gridcolor="#262932", zerolinecolor="#262932"),
    )
)

EXPECTED_COLS = [
    "Movie_ID", "Title", "Genre", "Release_Year", "Duration", "Rating",
    "IMDb_Rating", "Votes", "Budget", "Revenue", "Date_Added", "Country",
    "Language", "Director", "Cast", "Production_House",
]

# ----------------------------------------------------------------------------
# SAMPLE "DIRTY" DATA GENERATOR  (used when no CSV is uploaded, so the app
# is instantly demoable — handy when sharing the link on LinkedIn/Naukri)
# ----------------------------------------------------------------------------
def make_dirty_sample(n=400, seed=42):
    rng = np.random.default_rng(seed)

    titles = [
        "Inception", "the dark knight   ", "PARASITE!!!", "hell or high water !!!!",
        "His House (Extended Collector's Edition with Bonus Features)",
        "Coco\n", "  Amélie", "Whiplash", "ROMA", "The Irishman", "Marriage Story",
        "1917", "Soul", "Nomadland", "Minari", "Tenet", "Dune", "Encanto",
        "The Power of the Dog", "CODA!!!", "  Everything Everywhere All at Once",
    ]
    genres_raw = [
        "Action", "action", "ACTION", "Drama, Thriller", "Drama | Comedy",
        "Comedy/Romance", " Fantasy", "\tDrama", "Horror", "sci-fi", "Sci-Fi",
        "Documentary", "Animation", "Crime, Drama", np.nan,
    ]
    countries_raw = ["USA", "US", "U.S.", "United States", "U.S.A", "UK", "U.K.",
                      "United Kingdom", "Britain", "India", "South Korea", "France",
                      "Germany", "Japan", np.nan]
    langs_raw = ["english", "English", "ENGLISH", "Eng", "Hin", "Kor", "Ger",
                 "Fre", "Jap", "Spa", "Ita", "Chinese", np.nan]
    durations_raw = ["2h 28m", "148 mins", "132min", "1h 46m", "not available",
                      "95", "3h 1m", np.nan]
    dates_raw = ["2024-05-01", "01/05/2024", "May 1, 2024", "20240501",
                 "2021-11-19", "19/11/2021", np.nan]

    rows = []
    for i in range(n):
        mid = f"NF{1000 + i}"
        if i % 37 == 0 and i > 0:
            mid = f"NF{1000 + i - 1}"  # inject duplicate id

        year = int(rng.choice([rng.integers(1990, 2026), 1890, 2099], p=[0.94, 0.03, 0.03]))
        rating = rng.choice([str(round(rng.uniform(1, 10), 1)), "", "N/A"], p=[0.9, 0.06, 0.04])
        imdb = round(float(rng.choice([rng.uniform(0, 10), rng.uniform(10, 15), -1], p=[0.92, 0.04, 0.04])), 1)
        votes = rng.choice([f"{rng.integers(1000, 2000000):,}", "", "unknown"], p=[0.9, 0.06, 0.04])
        budget = rng.choice([f"${rng.integers(1_000_000, 250_000_000):,}", f"-{rng.integers(1000,500000)}", ""], p=[0.88, 0.06, 0.06])
        revenue = rng.choice([f"{rng.integers(0, 900_000_000):,}", "0", ""], p=[0.85, 0.08, 0.07])

        rows.append({
            "Movie_ID": mid,
            "Title": rng.choice(titles),
            "Genre": rng.choice(genres_raw),
            "Release_Year": year,
            "Duration": rng.choice(durations_raw),
            "Rating": rating,
            "IMDb_Rating": imdb,
            "Votes": votes,
            "Budget": budget,
            "Revenue": revenue,
            "Date_Added": rng.choice(dates_raw),
            "Country": rng.choice(countries_raw),
            "Language": rng.choice(langs_raw),
            "Director": rng.choice(["Chris Nolan", "Bong Joon-ho", np.nan, "Greta Gerwig", "Denis Villeneuve"]),
            "Cast": rng.choice(["Actor A, Actor B", np.nan, "Actor C, Actor D, Actor E"]),
            "Production_House": rng.choice(["Netflix Studios", np.nan, "A24", "Warner Bros"]),
        })

    df = pd.DataFrame(rows)
    # blank out a few titles entirely to simulate missing titles
    blank_idx = rng.choice(df.index, size=6, replace=False)
    df.loc[blank_idx, "Title"] = np.nan
    return df


# ----------------------------------------------------------------------------
# CLEANING FUNCTIONS  (corrected + hardened versions of the notebook logic)
# ----------------------------------------------------------------------------
COUNTRY_MAP = {
    "u.s.": "United States", "us": "United States", "usa": "United States",
    "united states": "United States", "u.s.a": "United States",
    "u.k.": "United Kingdom", "uk": "United Kingdom",
    "united kingdom": "United Kingdom", "britain": "United Kingdom",
}
LANGUAGE_MAP = {
    "english": "English", "eng": "English",
    "hin": "Hindi", "hindi": "Hindi",
    "kor": "Korean", "korean": "Korean",
    "ger": "German", "german": "German",
    "fre": "French", "french": "French",
    "jap": "Japanese", "japanese": "Japanese",
    "spa": "Spanish", "spanish": "Spanish",
    "ita": "Italian", "italian": "Italian",
    "chinese": "Chinese",
}


def clean_title(title):
    if pd.isnull(title):
        return np.nan
    title = str(title).replace("\n", " ").replace("\t", " ")
    title = " ".join(title.split())
    title = title.rstrip("!").strip()
    if "(" in title:
        title = title[: title.index("(")].strip()
    title = title.title()
    return title if title else np.nan


def clean_genre(genre):
    if pd.isnull(genre):
        return np.nan
    genre = str(genre).replace("\n", " ").replace("\t", " ")
    genre = genre.replace("|", ",").replace("/", ",")
    genre = " ".join(genre.split())
    genre = genre.split(",")[0].strip()
    return genre.title() if genre else np.nan


def clean_year(year, lower=1900, upper=2025):
    if pd.isnull(year):
        return np.nan
    try:
        year = float(year)
    except (ValueError, TypeError):
        return np.nan
    if year < lower or year > upper:
        return np.nan
    return year


def clean_duration(duration):
    if pd.isnull(duration):
        return np.nan
    d = str(duration).strip().lower()
    try:
        if "h" in d:
            match = re.match(r"(\d+)\s*h\s*(\d*)\s*m?", d)
            if not match:
                return np.nan
            hours = int(match.group(1))
            minutes = int(match.group(2)) if match.group(2) else 0
            total = hours * 60 + minutes
        elif "min" in d:
            total = int(re.sub(r"[^\d]", "", d))
        elif d.isdigit():
            total = int(d)
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan
    if total < 30 or total > 600:
        return np.nan
    return total


def clean_rating(rating):
    return pd.to_numeric(rating, errors="coerce")


def clean_imdb(rating):
    val = pd.to_numeric(rating, errors="coerce")
    if pd.isnull(val) or val > 10 or val < 0:
        return np.nan
    return val


def clean_votes(votes):
    if pd.isnull(votes):
        return np.nan
    v = "".join(str(votes).split(","))
    return int(v) if v.isdigit() else np.nan


def clean_budget(budget):
    if pd.isnull(budget):
        return np.nan
    b = str(budget).replace("$", "").replace(",", "").strip()
    try:
        b = float(b)
    except ValueError:
        return np.nan
    return np.nan if b < 0 else b


def clean_revenue(revenue):
    if pd.isnull(revenue):
        return np.nan
    r = str(revenue).replace(",", "").strip()
    try:
        r = float(r)
    except ValueError:
        return np.nan
    return np.nan if r <= 0 else r


def clean_country(value):
    if pd.isnull(value):
        return np.nan
    return COUNTRY_MAP.get(str(value).strip().lower(), str(value).strip())


def clean_language(value):
    if pd.isnull(value):
        return np.nan
    return LANGUAGE_MAP.get(str(value).strip().lower(), str(value).strip())


def clean_text_generic(value):
    if pd.isnull(value):
        return np.nan
    v = str(value).strip()
    return np.nan if v == "" else v


def parse_dates(series):
    s = series.astype(str)
    compact = s.str.match(r"^\d{8}$")
    s = s.where(~compact, s.str.replace(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True))
    return pd.to_datetime(s, format="mixed", errors="coerce")


STEP_DEFS = [
    dict(key="dup_id", label="Remove duplicate Movie IDs",
         issue="Rows sharing the same Movie_ID", fix="Kept first occurrence, dropped rest"),
    dict(key="title", label="Clean Title column",
         issue="Whitespace, newlines, ALL CAPS, trailing !!!, long subtitles", fix="Normalized + Title Case"),
    dict(key="missing_title", label="Drop rows with no title",
         issue="Blank/whitespace-only titles", fix="Row removed (unidentifiable movie)"),
    dict(key="genre", label="Standardise Genre",
         issue="Mixed casing & multi-genre strings (comma / pipe / slash)", fix="First genre kept, Title Case"),
    dict(key="year", label="Fix Release_Year outliers",
         issue="Years like 1890 or 2099", fix="Values outside 1900–2025 → NaN"),
    dict(key="duration", label="Parse Duration formats",
         issue="'2h 28m', '148 mins', '132min' mixed formats", fix="Converted to total minutes (int)"),
    dict(key="rating", label="Convert Rating to numeric",
         issue="Stored as text/strings", fix="pd.to_numeric(errors='coerce')"),
    dict(key="imdb", label="Fix IMDb_Rating range",
         issue="Values outside 0–10", fix="Out-of-range values → NaN"),
    dict(key="votes", label="Clean Votes column",
         issue="Comma-formatted numbers as text", fix="Commas stripped, cast to int"),
    dict(key="budget", label="Clean Budget column",
         issue="'$' signs, commas, negative values", fix="Symbols stripped, negatives → NaN"),
    dict(key="revenue", label="Clean Revenue column",
         issue="Commas, revenue = 0", fix="Symbols stripped, zero/negative → NaN"),
    dict(key="date", label="Standardise Date_Added",
         issue="4 different date formats incl. compact YYYYMMDD", fix="Unified to datetime via pd.to_datetime"),
    dict(key="country", label="Standardise Country names",
         issue="USA / US / U.S. / United States variants", fix="Mapped via lookup dictionary"),
    dict(key="language", label="Standardise Language",
         issue="Abbreviations + inconsistent casing", fix="Mapped via lookup dictionary"),
    dict(key="text_cols", label="Trim Director / Cast / Production House",
         issue="Stray whitespace, empty strings not treated as NaN", fix="Stripped + empty → NaN"),
    dict(key="near_dup", label="Drop near-duplicate rows",
         issue="Same Title + Release_Year appearing twice", fix="Kept first occurrence"),
    dict(key="fillna", label="Fill remaining missing numerics",
         issue="Duration / Rating / IMDb_Rating / Votes still missing", fix="Filled with column median"),
]


def run_pipeline(df_in: pd.DataFrame, enabled: dict) -> tuple[pd.DataFrame, list]:
    df = df_in.copy()
    log = []

    def track(step_key, label, fn):
        nonlocal df
        before_rows = len(df)
        before_missing = int(df.isnull().sum().sum())
        if enabled.get(step_key, True):
            fn()
        after_rows = len(df)
        after_missing = int(df.isnull().sum().sum())
        log.append(dict(
            label=label, applied=enabled.get(step_key, True),
            rows_removed=before_rows - after_rows,
            missing_delta=after_missing - before_missing,
        ))

    if "Movie_ID" in df.columns:
        track("dup_id", "Remove duplicate Movie IDs",
              lambda: df.drop_duplicates(subset="Movie_ID", keep="first", inplace=True))

    if "Title" in df.columns:
        track("title", "Clean Title column",
              lambda: df.__setitem__("Title", df["Title"].apply(clean_title)))
        track("missing_title", "Drop rows with no title",
              lambda: df.dropna(subset=["Title"], inplace=True))

    if "Genre" in df.columns:
        track("genre", "Standardise Genre",
              lambda: df.__setitem__("Genre", df["Genre"].apply(clean_genre)))

    if "Release_Year" in df.columns:
        track("year", "Fix Release_Year outliers",
              lambda: df.__setitem__("Release_Year", df["Release_Year"].apply(clean_year)))

    if "Duration" in df.columns:
        track("duration", "Parse Duration formats",
              lambda: df.__setitem__("Duration", df["Duration"].apply(clean_duration)))

    if "Rating" in df.columns:
        track("rating", "Convert Rating to numeric",
              lambda: df.__setitem__("Rating", clean_rating(df["Rating"])))

    if "IMDb_Rating" in df.columns:
        track("imdb", "Fix IMDb_Rating range",
              lambda: df.__setitem__("IMDb_Rating", df["IMDb_Rating"].apply(clean_imdb)))

    if "Votes" in df.columns:
        track("votes", "Clean Votes column",
              lambda: df.__setitem__("Votes", df["Votes"].apply(clean_votes)))

    if "Budget" in df.columns:
        track("budget", "Clean Budget column",
              lambda: df.__setitem__("Budget", df["Budget"].apply(clean_budget)))

    if "Revenue" in df.columns:
        track("revenue", "Clean Revenue column",
              lambda: df.__setitem__("Revenue", df["Revenue"].apply(clean_revenue)))

    if "Date_Added" in df.columns:
        track("date", "Standardise Date_Added",
              lambda: df.__setitem__("Date_Added", parse_dates(df["Date_Added"])))

    if "Country" in df.columns:
        track("country", "Standardise Country names",
              lambda: df.__setitem__("Country", df["Country"].apply(clean_country)))

    if "Language" in df.columns:
        track("language", "Standardise Language",
              lambda: df.__setitem__("Language", df["Language"].apply(clean_language)))

    text_cols = [c for c in ["Director", "Cast", "Production_House"] if c in df.columns]
    if text_cols:
        def _clean_text_cols():
            for c in text_cols:
                df[c] = df[c].apply(clean_text_generic)
        track("text_cols", "Trim Director / Cast / Production House", _clean_text_cols)

    if {"Title", "Release_Year"}.issubset(df.columns):
        track("near_dup", "Drop near-duplicate rows",
              lambda: df.drop_duplicates(subset=["Title", "Release_Year"], keep="first", inplace=True))

    def _fillna():
        for c in ["Duration", "Rating", "IMDb_Rating", "Votes"]:
            if c in df.columns and df[c].notnull().any():
                df[c] = df[c].fillna(df[c].median())
        if "Title" in df.columns:
            df["Title"] = df["Title"].fillna("Unknown")
    track("fillna", "Fill remaining missing numerics", _fillna)

    return df, log


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def metric_card(label, value, sub="", color=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color}">{value}</div>
            <div class="metric-delta">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def missing_bar_chart(df, title):
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    if miss.empty:
        return None
    fig = go.Figure(go.Bar(
        x=miss.values, y=miss.index, orientation="h",
        marker_color="#D6303C",
        text=miss.values, textposition="outside",
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], title=title, height=max(280, 28 * len(miss)))
    return fig


def to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "log" not in st.session_state:
    st.session_state.log = None

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 Data Source")
    uploaded = st.file_uploader("Upload netflix_movies_dirty.csv", type=["csv"])

    c1, c2 = st.columns(2)
    with c1:
        load_sample = st.button("▶ Use sample data", use_container_width=True)
    with c2:
        clear = st.button("✕ Clear", use_container_width=True)

    if clear:
        st.session_state.df_raw = None
        st.session_state.df_clean = None
        st.session_state.log = None

    if uploaded is not None:
        try:
            st.session_state.df_raw = pd.read_csv(uploaded, dtype=str)
            st.session_state.df_clean = None
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

    if load_sample:
        st.session_state.df_raw = make_dirty_sample()
        st.session_state.df_clean = None

    st.markdown("---")
    st.markdown("### 🧹 Pipeline Steps")
    st.caption("Toggle individual cleaning steps on/off and re-run.")

    enabled = {}
    for step in STEP_DEFS:
        enabled[step["key"]] = st.checkbox(step["label"], value=True, key=f"chk_{step['key']}")

    st.markdown("---")
    run_clicked = st.button("⚙ Run cleaning pipeline", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown('<div class="reel-eyebrow">PANDAS · NUMPY · DATA QUALITY ENGINEERING</div>', unsafe_allow_html=True)
st.markdown('<h1 class="reel-title">NETFLIX <span>DATA CLEANING</span> STUDIO</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="reel-sub">An interactive walkthrough of a real-world data-cleaning pipeline — '
    'missing values, duplicate IDs, inconsistent text, mixed date formats and out-of-range '
    'outliers — turned into a clean, analysis-ready dataset. Upload a CSV or load the sample '
    'dirty dataset to see every step run live.</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="reel-divider">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# EMPTY STATE
# ----------------------------------------------------------------------------
if st.session_state.df_raw is None:
    st.info("👈 Upload a CSV in the sidebar, or click **Use sample data** to try the app instantly.")
    st.stop()

df_raw = st.session_state.df_raw

if run_clicked or st.session_state.df_clean is None:
    with st.spinner("Running cleaning pipeline..."):
        df_clean, log = run_pipeline(df_raw, enabled)
        st.session_state.df_clean = df_clean
        st.session_state.log = log

df_clean = st.session_state.df_clean
log = st.session_state.log

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_overview, tab_quality, tab_pipeline, tab_compare, tab_download = st.tabs(
    ["📊 Overview", "🔍 Quality Report", "🧹 Pipeline", "📈 Before vs After", "💾 Download"]
)

# ---- OVERVIEW ---------------------------------------------------------------
with tab_overview:
    cols = st.columns(4)
    with cols[0]:
        metric_card("Raw rows", f"{df_raw.shape[0]:,}")
    with cols[1]:
        metric_card("Raw columns", f"{df_raw.shape[1]}")
    with cols[2]:
        missing_pct = df_raw.isnull().mean().mean() * 100
        metric_card("Missing cells", f"{missing_pct:.1f}%", color="red")
    with cols[3]:
        dup_ids = 0
        if "Movie_ID" in df_raw.columns:
            dup_ids = int(df_raw["Movie_ID"].duplicated().sum())
        metric_card("Duplicate IDs", f"{dup_ids}", color="amber")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.markdown("##### Raw data preview")
        st.dataframe(df_raw.head(20), use_container_width=True, height=380)
    with right:
        st.markdown("##### Column types")
        dtypes_df = pd.DataFrame({"column": df_raw.columns, "dtype": df_raw.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True, height=380, hide_index=True)

# ---- QUALITY REPORT ----------------------------------------------------------
with tab_quality:
    st.markdown("##### Missing values by column (raw)")
    fig = missing_bar_chart(df_raw, "")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected in the raw file.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n = int(df_raw["Movie_ID"].duplicated().sum()) if "Movie_ID" in df_raw.columns else 0
        metric_card("Duplicate Movie_IDs", n, color="red")
    with c2:
        n = 0
        if "Release_Year" in df_raw.columns:
            yrs = pd.to_numeric(df_raw["Release_Year"], errors="coerce")
            n = int(((yrs < 1900) | (yrs > 2025)).sum())
        metric_card("Release_Year outliers", n, color="amber")
    with c3:
        n = 0
        if "IMDb_Rating" in df_raw.columns:
            r = pd.to_numeric(df_raw["IMDb_Rating"], errors="coerce")
            n = int(((r < 0) | (r > 10)).sum())
        metric_card("IMDb_Rating out of range", n, color="amber")

# ---- PIPELINE -----------------------------------------------------------------
with tab_pipeline:
    st.markdown("##### Step-by-step pipeline log")
    for i, entry in enumerate(log, start=1):
        badge_class = "fixed" if entry["applied"] else "issue"
        badge_text = "APPLIED" if entry["applied"] else "SKIPPED"
        rows_txt = f"−{entry['rows_removed']} rows" if entry["rows_removed"] else "no rows removed"
        miss_txt = f"missing cells {entry['missing_delta']:+d}"
        with st.container():
            c1, c2 = st.columns([0.06, 0.94])
            with c1:
                st.markdown(f'<div class="step-num">{i}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"""<div class="card">
                        <span class="badge {badge_class}">{badge_text}</span>
                        <strong>{entry['label']}</strong>
                        <div class="metric-delta">{rows_txt} · {miss_txt}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---- BEFORE / AFTER -----------------------------------------------------------
with tab_compare:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows: raw → clean", f"{df_raw.shape[0]} → {df_clean.shape[0]}")
    with c2:
        raw_missing = int(df_raw.isnull().sum().sum())
        clean_missing = int(df_clean.isnull().sum().sum())
        metric_card("Missing cells", f"{raw_missing} → {clean_missing}", color="good")
    with c3:
        dup_raw = int(df_raw["Movie_ID"].duplicated().sum()) if "Movie_ID" in df_raw.columns else 0
        dup_clean = int(df_clean["Movie_ID"].duplicated().sum()) if "Movie_ID" in df_clean.columns else 0
        metric_card("Duplicate IDs", f"{dup_raw} → {dup_clean}", color="good")
    with c4:
        rows_removed = df_raw.shape[0] - df_clean.shape[0]
        pct = (rows_removed / df_raw.shape[0] * 100) if df_raw.shape[0] else 0
        metric_card("Rows removed", f"{rows_removed} ({pct:.1f}%)", color="amber")

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("##### Missing values — raw")
        fig = missing_bar_chart(df_raw, "")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values.")
    with m2:
        st.markdown("##### Missing values — cleaned")
        fig2 = missing_bar_chart(df_clean, "")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("No missing values.")

    if "IMDb_Rating" in df_clean.columns:
        st.markdown("##### IMDb rating distribution (cleaned)")
        vals = pd.to_numeric(df_clean["IMDb_Rating"], errors="coerce").dropna()
        if len(vals):
            fig3 = go.Figure(go.Histogram(x=vals, marker_color="#D6303C", nbinsx=20))
            fig3.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("##### Cleaned data preview")
    st.dataframe(df_clean.head(20), use_container_width=True, height=380)

# ---- DOWNLOAD -----------------------------------------------------------------
with tab_download:
    st.markdown("##### Export the cleaned dataset")
    st.write(f"Final shape: **{df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns**")
    st.download_button(
        "⬇ Download cleaned CSV",
        data=to_csv_bytes(df_clean),
        file_name=f"netflix_movies_cleaned_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("##### Pipeline summary")
    summary_df = pd.DataFrame(log)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown(
    '<footer class="credit">Built with Streamlit · Pandas · NumPy · Plotly &nbsp;—&nbsp; '
    'a portfolio project demonstrating real-world data cleaning workflows.</footer>',
    unsafe_allow_html=True,
)