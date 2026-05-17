"""
app.py — Streamlit dashboard for the RAG Architecture Comparison Experiment.

Run with:
    streamlit run app.py

Four tabs:
  1. Overview    — summary stats + winner table
  2. Charts      — all 4 visualisations
  3. Browse      — searchable Q&A results per architecture
  4. Architecture deep-dive — per-architecture score distributions
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Comparison Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
ARCH_COLORS = {
    "vanilla_rag": "#4C72B0",
    "hyde_rag":    "#DD8452",
    "graph_rag":   "#55A868",
    "agentic_rag": "#C44E52",
}
ARCH_LABELS = {
    "vanilla_rag": "Vanilla RAG",
    "hyde_rag":    "HyDE RAG",
    "graph_rag":   "Graph RAG",
    "agentic_rag": "Agentic RAG",
}
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "judge_factuality":  "Judge Factuality",
    "judge_relevance":   "Judge Relevance",
    "judge_citation":    "Judge Citation",
    "latency_ms":        "Latency (ms)",
}

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .winner-badge {
        background: #f0c040;
        color: #111;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .arch-pill {
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    if not config.RESULTS_FILE.exists():
        return None
    df = pd.read_csv(config.RESULTS_FILE)
    if "question_id" not in df.columns:
        q_map = {q: i for i, q in enumerate(df["question"].unique())}
        df["question_id"] = df["question"].map(q_map)
    return df

@st.cache_data
def load_papers():
    if not config.PAPERS_FILE.exists():
        return []
    import json
    with open(config.PAPERS_FILE) as f:
        return json.load(f)

@st.cache_data
def load_questions():
    if not config.QUESTIONS_FILE.exists():
        return []
    import json
    with open(config.QUESTIONS_FILE) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 RAG Comparison")
    st.caption("Hallucination Reduction Study")
    st.divider()

    df_full = load_data()

    if df_full is not None:
        archs = sorted(df_full["architecture"].unique())
        selected_archs = st.multiselect(
            "Architectures to show",
            options=archs,
            default=archs,
            format_func=lambda x: ARCH_LABELS.get(x, x),
        )

        ragas_metrics   = ["faithfulness", "answer_relevancy", "context_precision"]
        judge_metrics   = ["judge_factuality", "judge_relevance", "judge_citation"]
        all_metrics     = ragas_metrics + judge_metrics + ["latency_ms"]

        selected_metric = st.selectbox(
            "Primary metric (for sorting)",
            options=all_metrics,
            format_func=lambda x: METRIC_LABELS.get(x, x),
        )

        st.divider()
        st.markdown("**Dataset**")
        papers    = load_papers()
        questions = load_questions()
        st.metric("Papers indexed", len(papers))
        st.metric("Eval questions", len(questions))
        st.metric("Total results rows", len(df_full))

        st.divider()
        if st.button("♻️ Refresh data"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.warning("No results yet.\n\nRun `python main.py` first.")
        selected_archs = []
        selected_metric = "faithfulness"


# ─────────────────────────────────────────────────────────────────────────────
# Guard — no data yet
# ─────────────────────────────────────────────────────────────────────────────

if df_full is None:
    st.title("🔬 RAG Architecture Comparison Dashboard")
    st.info(
        "No results file found at `data/rag_comparison_results.csv`.\n\n"
        "**To generate results:**\n"
        "```bash\n"
        "export OPENAI_API_KEY=sk-...\n"
        "python main.py --dry-run   # quick test (2 questions)\n"
        "python main.py             # full experiment\n"
        "```"
    )
    st.stop()

df = df_full[df_full["architecture"].isin(selected_archs)].copy()

# Normalise judge scores to 0-1 for radar chart
df_norm = df.copy()
for col in ["judge_factuality", "judge_relevance", "judge_citation"]:
    df_norm[col] = (df_norm[col] - 1) / 4


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_charts, tab_browse, tab_dive = st.tabs([
    "📊 Overview", "📈 Charts", "🔍 Browse Results", "🏗️ Deep Dive"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Experiment Overview")

    ragas_cols   = ["faithfulness", "answer_relevancy", "context_precision"]
    judge_cols   = ["judge_factuality", "judge_relevance", "judge_citation"]
    display_cols = ragas_cols + judge_cols + ["latency_ms"]

    summary = df.groupby("architecture")[display_cols].mean().round(3)
    std     = df.groupby("architecture")[display_cols].std().round(3)

    # ── Winner per metric ────────────────────────────────────────────────────
    st.subheader("Winner per metric")
    winner_cols = st.columns(len(display_cols))
    for col_widget, metric in zip(winner_cols, display_cols):
        with col_widget:
            if metric == "latency_ms":
                winner = summary[metric].idxmin()
                best   = f"{summary[metric].min():.0f} ms"
            else:
                winner = summary[metric].idxmax()
                best   = f"{summary[metric].max():.3f}"
            color  = ARCH_COLORS.get(winner, "#999")
            label  = ARCH_LABELS.get(winner, winner)
            st.markdown(
                f"**{METRIC_LABELS.get(metric, metric)}**\n\n"
                f"<span style='background:{color};color:white;border-radius:6px;"
                f"padding:3px 10px;font-size:0.8rem'>{label}</span><br>"
                f"<span style='font-size:1.1rem;font-weight:bold'>{best}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Summary table ────────────────────────────────────────────────────────
    st.subheader("Mean scores per architecture")

    def style_table(df_s):
        styled = df_s.style
        for metric in ragas_cols + judge_cols:
            if metric in df_s.columns:
                styled = styled.background_gradient(
                    subset=[metric], cmap="RdYlGn", vmin=0,
                    vmax=1 if metric in ragas_cols else 5
                )
        if "latency_ms" in df_s.columns:
            styled = styled.background_gradient(
                subset=["latency_ms"], cmap="RdYlGn_r"
            )
        return styled.format(
            {m: "{:.3f}" for m in ragas_cols}
            | {m: "{:.2f}" for m in judge_cols}
            | {"latency_ms": "{:.0f}"}
        )

    display_df = summary.copy()
    display_df.index = [ARCH_LABELS.get(i, i) for i in display_df.index]
    display_df.columns = [METRIC_LABELS.get(c, c) for c in display_df.columns]
    st.dataframe(style_table(display_df), use_container_width=True)

    st.caption("Green = better. Latency: lower is better. All other metrics: higher is better.")

    # ── Radar chart (Plotly, interactive) ───────────────────────────────────
    st.subheader("All metrics at a glance")

    radar_metrics = ["faithfulness", "answer_relevancy", "context_precision",
                     "judge_factuality", "judge_relevance", "judge_citation"]
    radar_labels  = ["Faithfulness", "Answer Relevancy", "Context Precision",
                     "Judge Factuality", "Judge Relevance", "Judge Citation"]

    radar_summary = df_norm.groupby("architecture")[radar_metrics].mean()
    fig_radar     = go.Figure()

    def hex_to_rgba(hex_color, alpha=0.1):
        """Convert #RRGGBB to rgba(r,g,b,alpha) for Plotly fillcolor."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for arch in radar_summary.index:
        vals  = radar_summary.loc[arch, radar_metrics].tolist()
        vals += vals[:1]
        color = ARCH_COLORS.get(arch, "#999999")
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.1),
            line=dict(color=color, width=2),
            name=ARCH_LABELS.get(arch, arch),
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=480,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Charts
# ══════════════════════════════════════════════════════════════════════════════

with tab_charts:
    st.header("Visualisations")

    chart_dir = config.VIZ_DIR
    saved_charts = {
        "faithfulness_bar.png":     "Faithfulness Score by Architecture",
        "latency_bar.png":          "Average Latency by Architecture",
        "radar_chart.png":          "All Metrics — Radar Chart",
        "faithfulness_heatmap.png": "Faithfulness Heatmap (Questions × Architectures)",
    }

    # Show pre-saved PNGs if they exist, otherwise render live
    png_found = any((chart_dir / name).exists() for name in saved_charts)

    if png_found:
        col1, col2 = st.columns(2)
        items = list(saved_charts.items())
        for i, (fname, title) in enumerate(items):
            path = chart_dir / fname
            col = col1 if i % 2 == 0 else col2
            with col:
                st.subheader(title)
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.info(f"{fname} not generated yet. Run `python visualizations/charts.py`.")
    else:
        st.info("PNG charts not found. Rendering live from current data …")

    st.divider()

    # ── Live faithfulness bar (always rendered from current filtered data) ──
    st.subheader("Live: Faithfulness by Architecture (filtered selection)")

    summary_live = df.groupby("architecture")["faithfulness"].agg(["mean", "std"]).reset_index()
    fig_bar, ax = plt.subplots(figsize=(8, 4))
    colors = [ARCH_COLORS.get(a, "#999") for a in summary_live["architecture"]]
    labels = [ARCH_LABELS.get(a, a) for a in summary_live["architecture"]]
    bars = ax.bar(labels, summary_live["mean"], yerr=summary_live["std"],
                  color=colors, capsize=5, edgecolor="white", width=0.6)
    for bar, mean in zip(bars, summary_live["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
    ax.set_facecolor("#0e1117"); fig_bar.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white"); ax.yaxis.label.set_color("white")
    ax.set_ylabel("Mean Faithfulness (0–1)", color="white")
    ax.set_ylim(0, 1.15)
    ax.spines[["top","right","left","bottom"]].set_color("#333")
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # ── Live latency bar ────────────────────────────────────────────────────
    st.subheader("Live: Latency by Architecture")

    lat_summary = df.groupby("architecture")["latency_ms"].agg(["mean", "std"]).reset_index()
    fig_lat, ax2 = plt.subplots(figsize=(8, 4))
    colors2 = [ARCH_COLORS.get(a, "#999") for a in lat_summary["architecture"]]
    labels2 = [ARCH_LABELS.get(a, a) for a in lat_summary["architecture"]]
    bars2 = ax2.bar(labels2, lat_summary["mean"], yerr=lat_summary["std"],
                    color=colors2, capsize=5, edgecolor="white", width=0.6)
    for bar, mean in zip(bars2, lat_summary["mean"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{mean:.0f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
    ax2.set_facecolor("#0e1117"); fig_lat.patch.set_facecolor("#0e1117")
    ax2.tick_params(colors="white"); ax2.yaxis.label.set_color("white")
    ax2.set_ylabel("Mean Latency (ms)", color="white")
    ax2.spines[["top","right","left","bottom"]].set_color("#333")
    st.pyplot(fig_lat, use_container_width=True)
    plt.close(fig_lat)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Browse Results
# ══════════════════════════════════════════════════════════════════════════════

with tab_browse:
    st.header("Browse Q&A Results")

    # ── Filters ──────────────────────────────────────────────────────────────
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
    with filter_col1:
        search_query = st.text_input("Search questions", placeholder="Type to filter questions …")
    with filter_col2:
        arch_filter = st.multiselect(
            "Architecture",
            options=sorted(df["architecture"].unique()),
            default=sorted(df["architecture"].unique()),
            format_func=lambda x: ARCH_LABELS.get(x, x),
        )
    with filter_col3:
        sort_by = st.selectbox("Sort by", ["question_id", "faithfulness", "latency_ms"],
                               format_func=lambda x: METRIC_LABELS.get(x, x))

    filtered = df[df["architecture"].isin(arch_filter)]
    if search_query:
        filtered = filtered[filtered["question"].str.contains(search_query, case=False, na=False)]
    filtered = filtered.sort_values(sort_by, ascending=(sort_by == "latency_ms"))

    st.caption(f"Showing {len(filtered)} rows")

    # ── Question-by-question comparison ─────────────────────────────────────
    questions_list = df["question"].unique().tolist()
    selected_q = st.selectbox(
        "Select a question to compare architectures side-by-side",
        options=questions_list,
        format_func=lambda q: q[:100] + ("…" if len(q) > 100 else ""),
    )

    if selected_q:
        q_rows = df[df["question"] == selected_q].sort_values("architecture")
        st.markdown(f"**Question:** {selected_q}")

        q_cols = st.columns(len(q_rows))
        for col_w, (_, row) in zip(q_cols, q_rows.iterrows()):
            arch  = row["architecture"]
            color = ARCH_COLORS.get(arch, "#999")
            label = ARCH_LABELS.get(arch, arch)
            with col_w:
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding:8px 12px;"
                    f"background:#1a1a2e;border-radius:4px'>"
                    f"<b style='color:{color}'>{label}</b><br><br>"
                    f"<small>Faithfulness: <b>{row['faithfulness']:.3f}</b><br>"
                    f"Relevancy: <b>{row['answer_relevancy']:.3f}</b><br>"
                    f"Latency: <b>{row['latency_ms']:.0f} ms</b></small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Full answer"):
                    st.write(row.get("answer", "—"))
                    if row.get("hyp_answer"):
                        st.caption(f"HyDE hypothetical: {row['hyp_answer']}")

    st.divider()

    # ── Full table ────────────────────────────────────────────────────────────
    st.subheader("Full results table")
    display_cols_browse = ["question_id", "architecture", "question",
                           "faithfulness", "answer_relevancy", "context_precision",
                           "judge_factuality", "judge_relevance", "judge_citation",
                           "latency_ms"]
    available = [c for c in display_cols_browse if c in filtered.columns]
    show_df = filtered[available].copy()
    show_df["architecture"] = show_df["architecture"].map(lambda x: ARCH_LABELS.get(x, x))
    show_df["question"]     = show_df["question"].str[:80] + "…"
    st.dataframe(show_df, use_container_width=True, height=400)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download filtered results as CSV",
        data=csv_bytes,
        file_name="rag_results_filtered.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Architecture Deep Dive
# ══════════════════════════════════════════════════════════════════════════════

with tab_dive:
    st.header("Architecture Deep Dive")

    chosen_arch = st.selectbox(
        "Choose architecture to inspect",
        options=sorted(df["architecture"].unique()),
        format_func=lambda x: ARCH_LABELS.get(x, x),
    )

    arch_df = df[df["architecture"] == chosen_arch]
    color   = ARCH_COLORS.get(chosen_arch, "#999")
    label   = ARCH_LABELS.get(chosen_arch, chosen_arch)

    st.markdown(
        f"<h3 style='color:{color}'>{label}</h3>",
        unsafe_allow_html=True,
    )

    # ── Top-level stats ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg Faithfulness",   f"{arch_df['faithfulness'].mean():.3f}")
    m2.metric("Avg Ans. Relevancy", f"{arch_df['answer_relevancy'].mean():.3f}")
    m3.metric("Avg Ctx Precision",  f"{arch_df['context_precision'].mean():.3f}")
    m4.metric("Avg Judge Score",
              f"{(arch_df[['judge_factuality','judge_relevance','judge_citation']].mean().mean()):.2f}/5")
    m5.metric("Avg Latency",        f"{arch_df['latency_ms'].mean():.0f} ms")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Faithfulness distribution ─────────────────────────────────────────────
    with col_left:
        st.subheader("Faithfulness distribution")
        fig_hist, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(arch_df["faithfulness"].dropna(), bins=10, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(arch_df["faithfulness"].mean(), color="yellow", linestyle="--", label="mean")
        ax.set_xlabel("Faithfulness Score", color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_facecolor("#0e1117"); fig_hist.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_color("#333")
        ax.legend(fontsize=9)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

    # ── Latency distribution ──────────────────────────────────────────────────
    with col_right:
        st.subheader("Latency distribution")
        fig_lat2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.hist(arch_df["latency_ms"].dropna(), bins=10, color=color, edgecolor="white", alpha=0.85)
        ax2.axvline(arch_df["latency_ms"].mean(), color="yellow", linestyle="--", label="mean")
        ax2.set_xlabel("Latency (ms)", color="white")
        ax2.set_ylabel("Count", color="white")
        ax2.set_facecolor("#0e1117"); fig_lat2.patch.set_facecolor("#0e1117")
        ax2.tick_params(colors="white")
        ax2.spines[["top","right","left","bottom"]].set_color("#333")
        ax2.legend(fontsize=9)
        st.pyplot(fig_lat2, use_container_width=True)
        plt.close(fig_lat2)

    # ── Per-question faithfulness line chart ─────────────────────────────────
    st.subheader("Faithfulness per question")
    q_faith = arch_df.sort_values("question_id")[["question_id", "faithfulness"]].reset_index(drop=True)
    fig_line, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(q_faith["question_id"], q_faith["faithfulness"], color=color, marker="o", ms=5, lw=1.5)
    ax3.axhline(q_faith["faithfulness"].mean(), color="yellow", linestyle="--", alpha=0.7, label="mean")
    ax3.fill_between(q_faith["question_id"], q_faith["faithfulness"], alpha=0.15, color=color)
    ax3.set_xlabel("Question #", color="white")
    ax3.set_ylabel("Faithfulness", color="white")
    ax3.set_ylim(0, 1.05)
    ax3.set_facecolor("#0e1117"); fig_line.patch.set_facecolor("#0e1117")
    ax3.tick_params(colors="white")
    ax3.spines[["top","right","left","bottom"]].set_color("#333")
    ax3.legend(fontsize=9)
    st.pyplot(fig_line, use_container_width=True)
    plt.close(fig_line)

    # ── Agentic tool call log ─────────────────────────────────────────────────
    if chosen_arch == "agentic_rag" and "tool_calls" in arch_df.columns:
        st.subheader("Tool call log (Agentic RAG)")
        import json as _json
        tool_call_counts = []
        for raw in arch_df["tool_calls"].dropna():
            try:
                calls = _json.loads(raw)
                tool_call_counts.append(len(calls))
            except Exception:
                tool_call_counts.append(0)
        avg_calls = sum(tool_call_counts) / max(len(tool_call_counts), 1)
        st.metric("Avg tool calls per question", f"{avg_calls:.1f}")

        # Show first row's tool calls as an example
        sample_raw = arch_df["tool_calls"].dropna().iloc[0] if len(arch_df) > 0 else "[]"
        try:
            sample_calls = _json.loads(sample_raw)
            if sample_calls:
                st.caption("Example tool call sequence (first question):")
                for tc in sample_calls:
                    st.markdown(
                        f"- **Step {tc.get('step','?')}**: `{tc.get('action','?')}` "
                        f"← `{str(tc.get('input',''))[:80]}`"
                    )
        except Exception:
            pass

    # ── Worst questions ───────────────────────────────────────────────────────
    st.subheader("Lowest faithfulness questions (hardest for this architecture)")
    worst = arch_df.nsmallest(5, "faithfulness")[["question", "faithfulness", "answer_relevancy", "latency_ms", "answer"]]
    worst["question"] = worst["question"].str[:100] + "…"
    worst["answer"]   = worst["answer"].str[:120] + "…"
    st.dataframe(worst.reset_index(drop=True), use_container_width=True)
