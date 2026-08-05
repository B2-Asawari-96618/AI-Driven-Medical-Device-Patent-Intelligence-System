from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover - Streamlit fallback handles this at runtime.
    alt = None


APP_TITLE = "Medical Device Patent Intelligence"
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

DATA_DIR = SRC_DIR / "data" / "processed"
RESULTS_DIR = SRC_DIR / "results"

PATENT_TOPIC_ANALYSIS = DATA_DIR / "patent_topic_analysis.csv"
RAG_DOCUMENTS = DATA_DIR / "rag_documents.csv"
TOPIC_SUMMARY = RESULTS_DIR / "bertopic" / "topic_summary.csv"
TOPIC_FREQUENCY = RESULTS_DIR / "bertopic" / "topic_frequency.csv"
CITATION_RANKING = RESULTS_DIR / "citation_analysis" / "citation_influence_ranking.csv"
MODEL_COMPARISON = RESULTS_DIR / "trend_analysis" / "model_comparison.csv"
PROPHET_FORECAST = RESULTS_DIR / "trend_analysis" / "prophet_forecast.csv"
PATENT_GROWTH_IMAGE = RESULTS_DIR / "trend_analysis" / "patent_growth.png"

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "can",
    "has",
    "have",
    "into",
    "method",
    "methods",
    "system",
    "systems",
    "device",
    "devices",
    "medical",
    "using",
    "used",
    "data",
    "patient",
    "patients",
}


def configure_page(page_title: str = APP_TITLE) -> None:
    st.set_page_config(page_title=page_title, layout="wide")
    inject_css()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #20242a;
            --muted: #5f6975;
            --line: #d9dee5;
            --panel: #f7f9fb;
            --teal: #0d7f83;
            --coral: #bc4b51;
            --amber: #b7791f;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.86rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        .patent-result {
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            background: #ffffff;
        }

        .patent-result strong {
            color: var(--ink);
        }

        .muted {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .pill {
            display: inline-block;
            padding: 0.1rem 0.45rem;
            margin-right: 0.25rem;
            border-radius: 999px;
            background: #eef6f6;
            color: #145f62;
            font-size: 0.78rem;
            border: 1px solid #c7e3e4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def file_status(path: Path) -> str:
    if not path.exists():
        return "Missing"
    if path.stat().st_size == 0:
        return "Empty"
    return "Ready"


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


@st.cache_data(show_spinner=False)
def read_csv_cached(path_text: str, usecols: tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    kwargs: dict[str, object] = {
        "encoding": "utf-8-sig",
        "low_memory": False,
    }
    if usecols is not None:
        wanted = set(usecols)
        kwargs["usecols"] = lambda col: col in wanted
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading patent metadata...")
def load_patent_metadata() -> pd.DataFrame:
    usecols = (
        "patent_id",
        "title",
        "country",
        "assignee_en",
        "filing_year",
        "forward_citation_count",
        "backward_citation_count",
        "topic_name",
    )
    df = read_csv_cached(str(PATENT_TOPIC_ANALYSIS), usecols)
    if df.empty:
        return df

    df = df.rename(columns={"topic_name": "topic"})
    for col in ("filing_year", "forward_citation_count", "backward_citation_count", "topic"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "filing_year" in df.columns:
        df = df[df["filing_year"].notna()]
        df["filing_year"] = df["filing_year"].astype(int)

    text_cols = ("patent_id", "title", "country", "assignee_en")
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "topic" in df.columns:
        df["topic"] = df["topic"].astype("Int64")

    return df


@st.cache_data(show_spinner=False)
def load_topic_summary() -> pd.DataFrame:
    summary = read_csv_cached(str(TOPIC_SUMMARY))
    if summary.empty:
        summary = read_csv_cached(str(TOPIC_FREQUENCY))
    if summary.empty:
        return summary

    summary["Topic"] = pd.to_numeric(summary["Topic"], errors="coerce").astype("Int64")
    summary["Count"] = pd.to_numeric(summary["Count"], errors="coerce").fillna(0).astype(int)
    summary["keywords"] = summary["Representation"].apply(parse_keywords)
    summary["topic_label"] = summary.apply(
        lambda row: make_topic_label(row["Topic"], row.get("Name", ""), row["keywords"]),
        axis=1,
    )
    summary["short_label"] = summary["topic_label"].apply(lambda value: shorten(value, 48))
    return summary


@st.cache_data(show_spinner=False)
def load_citation_ranking() -> pd.DataFrame:
    usecols = (
        "source_url",
        "patent_id",
        "title",
        "title_en",
        "country",
        "assignee_en",
        "filing_year",
        "forward_citation_count",
        "backward_citation_count",
        "citation_influence_score",
    )
    df = read_csv_cached(str(CITATION_RANKING), usecols)
    if df.empty:
        return df

    for col in ("filing_year", "forward_citation_count", "backward_citation_count", "citation_influence_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["display_title"] = df.get("title_en", "").fillna("")
    if "title" in df.columns:
        df["display_title"] = df["display_title"].mask(df["display_title"].eq(""), df["title"].fillna(""))
    df["display_title"] = df["display_title"].apply(lambda value: shorten(value, 90))
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    df = read_csv_cached(str(MODEL_COMPARISON))
    for col in ("MAE", "RMSE", "MAPE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_prophet_forecast() -> pd.DataFrame:
    df = read_csv_cached(str(PROPHET_FORECAST))
    if df.empty:
        return df
    for col in ("yhat", "yhat_lower", "yhat_upper", "trend"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["ds"].apply(extract_year_from_forecast_date)
    return df[df["year"].notna()].copy()


@st.cache_data(show_spinner="Loading local patent text index...")
def load_assistant_index(source: str = "topic") -> pd.DataFrame:
    if source == "rag" and RAG_DOCUMENTS.exists():
        usecols = (
            "patent_id",
            "title_en",
            "country",
            "assignee_en",
            "filing_year",
            "forward_citation_count",
            "backward_citation_count",
            "text",
        )
        df = read_csv_cached(str(RAG_DOCUMENTS), usecols)
        df = df.rename(columns={"title_en": "title", "text": "document_text"})
    else:
        usecols = (
            "patent_id",
            "title",
            "country",
            "assignee_en",
            "filing_year",
            "forward_citation_count",
            "backward_citation_count",
            "topic_text",
            "topic_name",
        )
        df = read_csv_cached(str(PATENT_TOPIC_ANALYSIS), usecols)
        df = df.rename(columns={"topic_text": "document_text", "topic_name": "topic"})

    if df.empty:
        return df

    for col in ("filing_year", "forward_citation_count", "backward_citation_count", "topic"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("patent_id", "title", "country", "assignee_en", "document_text"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["document_text"] = (
        df["document_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 2500)
    )
    df["search_text"] = (
        df["title"].fillna("")
        + " "
        + df["assignee_en"].fillna("")
        + " "
        + df["country"].fillna("")
        + " "
        + df["document_text"].fillna("")
    ).str.lower()
    return df


def parse_keywords(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [part.strip(" '[]\"") for part in text.split(",") if part.strip(" '[]\"")]


def make_topic_label(topic: object, name: object, keywords: Iterable[str]) -> str:
    if pd.isna(topic):
        return "Unassigned"
    topic_int = int(topic)
    if topic_int == -1:
        return "Outlier / mixed documents"

    words = [
        clean_label_word(word)
        for word in keywords
        if clean_label_word(word) and clean_label_word(word) not in STOPWORDS
    ]
    if not words and name:
        words = [
            clean_label_word(word)
            for word in str(name).replace("-", "_").split("_")
            if clean_label_word(word) and clean_label_word(word) not in STOPWORDS and not word.isdigit()
        ]

    label = " / ".join(dict.fromkeys(words[:4]))
    if not label:
        label = "Technology cluster"
    return f"Topic {topic_int}: {label.title()}"


def clean_label_word(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9-]+", "", str(value).strip().lower())
    if len(text) < 3:
        return ""
    return text


def shorten(value: object, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def extract_year_from_forecast_date(value: object) -> float:
    text = str(value)
    date_match = re.match(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})", text)
    fractional_match = re.search(r"\.(?:0+)?(?P<year>\d{4})\b", text)
    if fractional_match:
        seed_year = int(fractional_match.group("year"))
        if 1900 <= seed_year <= 2100:
            if date_match:
                calendar_year = int(date_match.group("year"))
                month_day = f"{date_match.group('month')}-{date_match.group('day')}"
                if calendar_year >= 1970 and month_day != "01-01":
                    return float(seed_year + calendar_year - 1969)
            return float(seed_year)

    matches = [int(match) for match in re.findall(r"\d{4}", text)]
    plausible = [year for year in matches if 1900 <= year <= 2100]
    if plausible:
        return float(plausible[-1])
    try:
        parsed = pd.to_datetime(value)
        return float(parsed.year)
    except Exception:
        return np.nan


def metric_row(metadata: pd.DataFrame, topic_summary: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    patent_count = len(metadata) if not metadata.empty else 0
    country_count = metadata["country"].replace("", np.nan).nunique() if "country" in metadata else 0
    assignee_count = metadata["assignee_en"].replace("", np.nan).nunique() if "assignee_en" in metadata else 0
    topic_count = topic_summary[topic_summary["Topic"] != -1]["Topic"].nunique() if not topic_summary.empty else 0

    c1.metric("Patents", f"{patent_count:,}")
    c2.metric("Countries", f"{country_count:,}")
    c3.metric("Assignees", f"{assignee_count:,}")
    c4.metric("Discovered topics", f"{topic_count:,}")


def render_app() -> None:
    configure_page()

    metadata = load_patent_metadata()
    topic_summary = load_topic_summary()

    with st.sidebar:
        st.title(APP_TITLE)
        page = st.radio(
            "View",
            [
                "Dashboard",
                "Topic Explorer",
                "Citation Intelligence",
                "Trend Forecast",
                "Patent Search",
                "Patent Assistant",
                "Data Health",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Local project workspace")
        st.code(str(ROOT_DIR), language=None)

    if page == "Dashboard":
        render_dashboard(metadata, topic_summary)
    elif page == "Topic Explorer":
        render_topic_explorer(metadata, topic_summary)
    elif page == "Citation Intelligence":
        render_citation_intelligence(metadata)
    elif page == "Trend Forecast":
        render_trend_forecast(metadata)
    elif page == "Patent Search":
        render_patent_search(metadata, topic_summary)
    elif page == "Patent Assistant":
        render_assistant_page()
    else:
        render_data_health()


def render_dashboard(metadata: pd.DataFrame, topic_summary: pd.DataFrame) -> None:
    st.title(APP_TITLE)
    metric_row(metadata, topic_summary)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Patent Filing Trend")
        yearly = yearly_counts(metadata)
        if yearly.empty:
            st.info("Patent filing data is not available.")
        else:
            render_line_chart(yearly, "filing_year", "patent_count", "Filing year", "Patent count")

    with right:
        st.subheader("Top Countries")
        country_data = top_counts(metadata, "country", "patent_count", 12)
        if country_data.empty:
            st.info("Country data is not available.")
        else:
            render_bar_chart(country_data, "patent_count", "country", "Patent count", "Country", "#0d7f83")

    st.subheader("Largest Technology Topics")
    topics = topic_summary[topic_summary["Topic"] != -1].nlargest(18, "Count")
    if topics.empty:
        st.info("Topic data is not available.")
    else:
        render_bar_chart(topics, "Count", "short_label", "Patents", "Topic", "#bc4b51")

    st.subheader("Recent Patent Records")
    if metadata.empty:
        st.info("Patent records are not available.")
    else:
        recent = metadata.sort_values("filing_year", ascending=False).head(20)
        st.dataframe(
            display_patent_table(recent),
            hide_index=True,
            use_container_width=True,
        )


def render_topic_explorer(metadata: pd.DataFrame, topic_summary: pd.DataFrame) -> None:
    st.title("Topic Explorer")
    if metadata.empty or topic_summary.empty:
        st.info("Topic outputs are not available yet.")
        return

    topic_options = (
        topic_summary[topic_summary["Topic"] != -1]
        .sort_values("Count", ascending=False)
        [["Topic", "topic_label", "Count", "keywords"]]
        .copy()
    )
    topic_options["option"] = topic_options.apply(
        lambda row: f"{row['topic_label']} ({int(row['Count']):,})",
        axis=1,
    )

    selected_option = st.selectbox("Topic", topic_options["option"].tolist())
    selected_row = topic_options[topic_options["option"] == selected_option].iloc[0]
    topic_id = int(selected_row["Topic"])

    st.markdown(
        " ".join(f"<span class='pill'>{keyword}</span>" for keyword in selected_row["keywords"][:8]),
        unsafe_allow_html=True,
    )

    topic_patents = metadata[metadata["topic"] == topic_id].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Patents in topic", f"{len(topic_patents):,}")
    c2.metric(
        "Countries",
        f"{topic_patents['country'].replace('', np.nan).nunique():,}" if "country" in topic_patents else "0",
    )
    c3.metric(
        "Median filing year",
        f"{int(topic_patents['filing_year'].median())}" if len(topic_patents) else "-",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Topic Over Time")
        yearly = yearly_counts(topic_patents)
        if yearly.empty:
            st.info("No yearly data for this topic.")
        else:
            render_line_chart(yearly, "filing_year", "patent_count", "Filing year", "Patent count")

    with right:
        st.subheader("Leading Assignees")
        assignees = top_counts(topic_patents, "assignee_en", "patent_count", 12)
        if assignees.empty:
            st.info("No assignee data for this topic.")
        else:
            render_bar_chart(assignees, "patent_count", "assignee_en", "Patent count", "Assignee", "#b7791f")

    st.subheader("Patents")
    st.dataframe(display_patent_table(topic_patents.head(300)), hide_index=True, use_container_width=True)


def render_citation_intelligence(metadata: pd.DataFrame) -> None:
    st.title("Citation Intelligence")
    citation_df = load_citation_ranking()

    if citation_df.empty:
        st.info("Citation ranking output is not available.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Ranked patents", f"{len(citation_df):,}")
    c2.metric(
        "Top influence score",
        f"{citation_df['citation_influence_score'].max():.3f}"
        if citation_df["citation_influence_score"].notna().any()
        else "-",
    )
    c3.metric(
        "Max forward citations",
        f"{int(citation_df['forward_citation_count'].max()):,}"
        if citation_df["forward_citation_count"].notna().any()
        else "-",
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Most Influential Patents")
        top = citation_df.nlargest(15, "citation_influence_score").copy()
        top["label"] = top["patent_id"] + " - " + top["display_title"]
        render_bar_chart(top, "citation_influence_score", "label", "Influence score", "Patent", "#0d7f83")

    with right:
        st.subheader("Citation Mix")
        top = citation_df.nlargest(15, "citation_influence_score")
        stacked = top.melt(
            id_vars=["patent_id"],
            value_vars=["forward_citation_count", "backward_citation_count"],
            var_name="citation_type",
            value_name="count",
        )
        render_grouped_bar_chart(stacked, "patent_id", "count", "citation_type")

    if not metadata.empty:
        st.subheader("Citation Activity By Year")
        citation_year = (
            metadata.groupby("filing_year", as_index=False)
            .agg(
                avg_forward=("forward_citation_count", "mean"),
                avg_backward=("backward_citation_count", "mean"),
                patents=("patent_id", "count"),
            )
            .sort_values("filing_year")
        )
        citation_year = citation_year[citation_year["patents"] >= 3]
        metric = st.selectbox("Citation metric", ["avg_forward", "avg_backward"], format_func=format_metric_name)
        render_line_chart(citation_year, "filing_year", metric, "Filing year", format_metric_name(metric))

    st.subheader("Ranking Table")
    columns = [
        "patent_id",
        "display_title",
        "country",
        "assignee_en",
        "filing_year",
        "forward_citation_count",
        "backward_citation_count",
        "citation_influence_score",
        "source_url",
    ]
    available = [col for col in columns if col in citation_df.columns]
    st.dataframe(citation_df[available], hide_index=True, use_container_width=True)


def render_trend_forecast(metadata: pd.DataFrame) -> None:
    st.title("Trend Forecast")

    yearly = yearly_counts(metadata)
    forecast = load_prophet_forecast()
    comparison = load_model_comparison()

    c1, c2, c3 = st.columns(3)
    if not yearly.empty:
        peak = yearly.loc[yearly["patent_count"].idxmax()]
        c1.metric("First filing year", f"{int(yearly['filing_year'].min())}")
        c2.metric("Peak year", f"{int(peak['filing_year'])}", f"{int(peak['patent_count']):,} patents")
        c3.metric("Latest filing year", f"{int(yearly['filing_year'].max())}")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Historical Filings")
        if yearly.empty:
            st.info("Historical trend data is not available.")
        else:
            render_line_chart(yearly, "filing_year", "patent_count", "Filing year", "Patent count")

    with right:
        st.subheader("Model Error")
        if comparison.empty:
            st.info("Model comparison output is not available.")
        else:
            metric = st.selectbox("Error metric", ["MAE", "RMSE", "MAPE"])
            render_bar_chart(comparison, metric, "Model", metric, "Model", "#bc4b51")

    st.subheader("Prophet Forecast")
    if forecast.empty:
        st.info("Forecast output is not available.")
    else:
        forecast_view = forecast[["year", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_view["year"] = forecast_view["year"].astype(int)
        render_forecast_chart(forecast_view)
        st.dataframe(forecast_view.round(2), hide_index=True, use_container_width=True)

    if PATENT_GROWTH_IMAGE.exists() and PATENT_GROWTH_IMAGE.stat().st_size > 0:
        with st.expander("Saved chart image"):
            st.image(str(PATENT_GROWTH_IMAGE), use_container_width=True)


def render_patent_search(metadata: pd.DataFrame, topic_summary: pd.DataFrame) -> None:
    st.title("Patent Search")
    if metadata.empty:
        st.info("Patent metadata is not available.")
        return

    topic_lookup = topic_label_lookup(topic_summary)
    search, country, years = render_search_filters(metadata, topic_lookup)
    results = filter_metadata(metadata, search, country, years)

    st.metric("Matching patents", f"{len(results):,}")
    st.dataframe(display_patent_table(results.head(500), topic_lookup), hide_index=True, use_container_width=True)


def render_assistant_standalone() -> None:
    configure_page("Patent Assistant")
    render_assistant_page()


def render_assistant_page() -> None:
    st.title("Patent Assistant")

    with st.sidebar:
        source = st.radio(
            "Evidence source",
            ["topic", "rag"],
            format_func=lambda value: "Topic text index" if value == "topic" else "RAG documents",
        )
        top_n = st.slider("Evidence depth", 3, 12, 6)

    query = st.text_input(
        "Question",
        placeholder="Example: wearable ECG monitoring with deep learning alerts",
    )

    if not query.strip():
        st.stop()

    index = load_assistant_index(source)
    if index.empty:
        st.info("The local evidence index is not available.")
        return

    results = search_patents(index, query, top_n)
    if results.empty:
        st.warning("No matching local patent evidence found.")
        return

    render_evidence_answer(query, results)
    st.subheader("Evidence")
    for _, row in results.iterrows():
        render_result_card(row, query)


def render_search_filters(metadata: pd.DataFrame, topic_lookup: dict[int, str]) -> tuple[str, str, tuple[int, int]]:
    c1, c2, c3 = st.columns([1.5, 0.8, 1])
    with c1:
        search = st.text_input("Search", placeholder="Title, patent ID, assignee, or topic")
    with c2:
        countries = ["All"] + sorted([value for value in metadata["country"].dropna().unique() if value])
        country = st.selectbox("Country", countries)
    with c3:
        min_year = int(metadata["filing_year"].min())
        max_year = int(metadata["filing_year"].max())
        years = st.slider("Filing years", min_year, max_year, (min_year, max_year))
    return search, country, years


def filter_metadata(
    metadata: pd.DataFrame,
    search: str,
    country: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    results = metadata.copy()
    results = results[
        (results["filing_year"] >= years[0])
        & (results["filing_year"] <= years[1])
    ]
    if country != "All":
        results = results[results["country"] == country]

    if search.strip():
        query = search.strip().lower()
        text = (
            results["patent_id"].fillna("")
            + " "
            + results["title"].fillna("")
            + " "
            + results["assignee_en"].fillna("")
            + " "
            + results["country"].fillna("")
            + " "
            + results["topic"].astype(str)
        ).str.lower()
        results = results[text.str.contains(re.escape(query), na=False)]

    return results.sort_values(["filing_year", "forward_citation_count"], ascending=[False, False])


def search_patents(index: pd.DataFrame, query: str, top_n: int) -> pd.DataFrame:
    terms = tokenize(query)
    if not terms:
        return pd.DataFrame()

    score = np.zeros(len(index), dtype=float)
    search_text = index["search_text"]
    title_text = index["title"].str.lower()

    phrase = query.lower().strip()
    if len(phrase) >= 4:
        score += search_text.str.contains(re.escape(phrase), regex=True, na=False).to_numpy(dtype=float) * 4

    for term in terms:
        pattern = re.escape(term)
        score += search_text.str.contains(pattern, regex=True, na=False).to_numpy(dtype=float)
        score += title_text.str.contains(pattern, regex=True, na=False).to_numpy(dtype=float) * 1.5

    if "forward_citation_count" in index.columns:
        citations = pd.to_numeric(index["forward_citation_count"], errors="coerce").fillna(0)
        if citations.max() > 0:
            score += np.log1p(citations.to_numpy()) / np.log1p(citations.max())

    results = index.copy()
    results["score"] = score
    results = results[results["score"] > 0].sort_values("score", ascending=False).head(top_n)
    results["snippet"] = results["document_text"].apply(lambda value: make_snippet(value, terms))
    return results


def tokenize(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]{2,}", query.lower())
    return [token for token in dict.fromkeys(tokens) if token not in STOPWORDS]


def make_snippet(text: object, terms: list[str], width: int = 520) -> str:
    clean_text = re.sub(r"\s+", " ", str(text).strip())
    if not clean_text:
        return ""

    lower = clean_text.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    start = max(0, min(positions) - 120) if positions else 0
    end = min(len(clean_text), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    return prefix + clean_text[start:end].strip() + suffix


def render_evidence_answer(query: str, results: pd.DataFrame) -> None:
    top = results.head(3)
    assignees = top["assignee_en"].replace("", np.nan).dropna().head(3).tolist()
    countries = top["country"].replace("", np.nan).dropna().head(3).tolist()
    years = pd.to_numeric(top["filing_year"], errors="coerce").dropna()
    year_text = ""
    if not years.empty:
        year_text = f" Filing years in the strongest matches range from {int(years.min())} to {int(years.max())}."
    assignee_text = f" Leading assignees include {', '.join(dict.fromkeys(assignees))}." if assignees else ""
    country_text = f" Jurisdictions represented include {', '.join(dict.fromkeys(countries))}." if countries else ""

    st.subheader("Local Answer")
    st.write(
        f"For `{query}`, the local patent corpus returns {len(results)} high-signal evidence records."
        f"{year_text}{assignee_text}{country_text}"
    )
    st.dataframe(
        top[["patent_id", "title", "country", "assignee_en", "filing_year", "score"]].round({"score": 2}),
        hide_index=True,
        use_container_width=True,
    )


def render_result_card(row: pd.Series, query: str) -> None:
    title = shorten(row.get("title", ""), 120) or "Untitled patent"
    meta = " | ".join(
        str(value)
        for value in [
            row.get("patent_id", ""),
            row.get("country", ""),
            int(row["filing_year"]) if pd.notna(row.get("filing_year", np.nan)) else "",
            row.get("assignee_en", ""),
        ]
        if str(value).strip()
    )
    snippet = row.get("snippet", "")
    st.markdown(
        f"""
        <div class="patent-result">
            <strong>{escape_html(title)}</strong>
            <div class="muted">{escape_html(meta)} | score {float(row.get("score", 0)):.2f}</div>
            <div>{escape_html(snippet)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_patent_table(df: pd.DataFrame, topic_lookup: dict[int, str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy()
    if topic_lookup and "topic" in table.columns:
        table["topic_label"] = table["topic"].map(lambda value: topic_lookup.get(int(value), f"Topic {value}") if pd.notna(value) else "")
    for col in ("title", "assignee_en"):
        if col in table.columns:
            table[col] = table[col].apply(lambda value: shorten(value, 110))
    columns = [
        "patent_id",
        "title",
        "country",
        "assignee_en",
        "filing_year",
        "forward_citation_count",
        "backward_citation_count",
        "topic",
        "topic_label",
    ]
    return table[[col for col in columns if col in table.columns]]


def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "filing_year" not in df.columns:
        return pd.DataFrame()
    yearly = (
        df.dropna(subset=["filing_year"])
        .groupby("filing_year", as_index=False)
        .size()
        .rename(columns={"size": "patent_count"})
        .sort_values("filing_year")
    )
    return yearly


def top_counts(df: pd.DataFrame, column: str, value_name: str, limit: int) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    counts = (
        df[column]
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(limit)
        .rename_axis(column)
        .reset_index(name=value_name)
    )
    counts[column] = counts[column].apply(lambda value: shorten(value, 42))
    return counts


def topic_label_lookup(topic_summary: pd.DataFrame) -> dict[int, str]:
    if topic_summary.empty:
        return {}
    return {
        int(row["Topic"]): row["topic_label"]
        for _, row in topic_summary.dropna(subset=["Topic"]).iterrows()
    }


def render_line_chart(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> None:
    if alt is None:
        st.line_chart(df.set_index(x)[y])
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2, color="#0d7f83")
        .encode(
            x=alt.X(f"{x}:O", title=x_title),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=[x, alt.Tooltip(y, format=",.2f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


def render_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    color: str,
) -> None:
    if alt is None:
        st.bar_chart(df.set_index(y)[x])
        return
    chart = (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y(f"{y}:N", title=y_title, sort="-x"),
            tooltip=[y, alt.Tooltip(x, format=",.3f")],
        )
        .properties(height=max(300, min(620, 24 * len(df))))
    )
    st.altair_chart(chart, use_container_width=True)


def render_grouped_bar_chart(df: pd.DataFrame, x: str, y: str, color: str) -> None:
    if alt is None:
        pivot = df.pivot(index=x, columns=color, values=y).fillna(0)
        st.bar_chart(pivot)
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{y}:Q", title="Citation count"),
            y=alt.Y(f"{x}:N", sort="-x", title="Patent"),
            color=alt.Color(f"{color}:N", title="Type", scale=alt.Scale(range=["#0d7f83", "#bc4b51"])),
            tooltip=[x, color, alt.Tooltip(y, format=",.0f")],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def render_forecast_chart(df: pd.DataFrame) -> None:
    if alt is None:
        st.line_chart(df.set_index("year")["yhat"])
        return
    band = (
        alt.Chart(df)
        .mark_area(opacity=0.22, color="#9ac9c8")
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("yhat_lower:Q", title="Predicted patent count"),
            y2="yhat_upper:Q",
        )
    )
    line = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2.5, color="#0d7f83")
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("yhat:Q", title="Predicted patent count"),
            tooltip=[
                "year",
                alt.Tooltip("yhat", format=",.2f"),
                alt.Tooltip("yhat_lower", format=",.2f"),
                alt.Tooltip("yhat_upper", format=",.2f"),
            ],
        )
    )
    st.altair_chart((band + line).properties(height=390), use_container_width=True)


def render_data_health() -> None:
    st.title("Data Health")
    files = [
        PATENT_TOPIC_ANALYSIS,
        RAG_DOCUMENTS,
        TOPIC_SUMMARY,
        TOPIC_FREQUENCY,
        CITATION_RANKING,
        MODEL_COMPARISON,
        PROPHET_FORECAST,
        PATENT_GROWTH_IMAGE,
        RESULTS_DIR / "citation_analysis" / "citation_distribution.png",
        RESULTS_DIR / "citation_analysis" / "top_influential_patents.png",
    ]
    table = pd.DataFrame(
        {
            "file": [str(path.relative_to(ROOT_DIR)) for path in files],
            "status": [file_status(path) for path in files],
            "size_mb": [file_size_mb(path) for path in files],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    missing = table[table["status"] != "Ready"]
    if not missing.empty:
        st.warning("Some generated files are missing or empty. The app falls back to CSV-driven charts where possible.")


def format_metric_name(value: str) -> str:
    return value.replace("_", " ").title()


def escape_html(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

if __name__ == "__main__":
    render_app()