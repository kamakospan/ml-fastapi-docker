"""Streamlit frontend. Talks only to the FastAPI service (never loads models itself).

Run:  streamlit run ui/app.py
Env:  API_URL (default http://localhost:8000), MLFLOW_UI_URL (default http://localhost:5000)
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")
MALIGNANT, BENIGN = "#d62728", "#2ca02c"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

st.set_page_config(page_title="Breast Cancer ML System", page_icon="🩺", layout="wide")


# ------------------------------------------------------------------ API calls
def api(method: str, path: str, **kw):
    r = requests.request(method, f"{API_URL}{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def get_features():
    return api("GET", "/features")


@st.cache_data(ttl=60)
def get_models():
    return api("GET", "/models")


@st.cache_data(ttl=60)
def get_evaluation():
    return api("GET", "/models/evaluation")


NICE = {"svm": "SVM", "knn": "KNN"}


def pretty(name: str) -> str:
    return NICE.get(name, name.replace("_", " ").title())


# ------------------------------------------------------------------ connection check
try:
    health = api("GET", "/")
except Exception as e:
    st.error(f"Can't reach the API at {API_URL}. Start it first (`uvicorn app.main:app`).\n\n{e}")
    st.stop()

feat = get_features()
FEATURES, STATS = feat["features"], feat["stats"]
models = get_models()
champion = health["champion"]

# session state: one value per feature, starts at the training median
for f in FEATURES:
    st.session_state.setdefault(f"f_{f}", float(STATS[f]["median"]))
st.session_state.setdefault("true_label", None)
st.session_state.setdefault("result", None)


def set_features(values: dict, true_label=None):
    for f, v in values.items():
        st.session_state[f"f_{f}"] = float(v)
    st.session_state.true_label = true_label
    st.session_state.result = None


def load_sample(label):
    s = api("GET", "/sample", params={"label": label})
    set_features(s["features"], s["true_label"])


def reset_medians():
    set_features({f: STATS[f]["median"] for f in FEATURES})


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🩺 ML System")
    st.success(f"API online · {health['models_loaded']} models")
    st.caption(f"Source: {health['source']}")
    st.markdown(f"**Champion:** {pretty(champion)}")
    st.divider()
    options = ["champion"] + [m["model_type"] for m in models]
    chosen = st.selectbox(
        "Model for prediction", options,
        format_func=lambda o: f"⭐ Champion ({pretty(champion)})" if o == "champion"
        else f"{pretty(o)} · v{next(m['version'] for m in models if m['model_type'] == o)}",
    )
    st.divider()
    st.link_button("Open MLflow UI ↗", MLFLOW_UI_URL, width="stretch")
    st.link_button("Open API docs ↗", f"{os.getenv('API_PUBLIC_URL', API_URL)}/docs", width="stretch")

st.title("Breast Cancer Diagnosis — ML System")
st.caption("Train → MLflow tracking & registry → FastAPI serving → Streamlit UI. "
           "Positive class = **malignant**.")

tab_pred, tab_cmp, tab_reg = st.tabs(["🔮 Predict", "📊 Compare models", "🧪 MLflow registry"])

# ================================================================== PREDICT
with tab_pred:
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.button("Random malignant patient", on_click=load_sample, args=("malignant",), width="stretch")
    b2.button("Random benign patient", on_click=load_sample, args=("benign",), width="stretch")
    b3.button("Reset to medians", on_click=reset_medians, width="stretch")
    if st.session_state.true_label:
        st.info(f"Loaded a real test-set patient. True diagnosis: **{st.session_state.true_label}** "
                "(stops being meaningful once you edit the values).")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("Patient features")
        groups = {
            "Mean values": [f for f in FEATURES if f.startswith("mean ")],
            "Standard error": [f for f in FEATURES if f.endswith(" error")],
            "Worst (largest) values": [f for f in FEATURES if f.startswith("worst ")],
        }
        for gi, (gname, cols) in enumerate(groups.items()):
            with st.expander(gname, expanded=(gi == 0)):
                c1, c2 = st.columns(2)
                for i, f in enumerate(cols):
                    s = STATS[f]
                    step = (s["max"] - s["min"]) / 200
                    (c1 if i % 2 == 0 else c2).slider(
                        f, min_value=float(s["min"]), max_value=float(s["max"]),
                        step=float(step), key=f"f_{f}", format="%.4f",
                    )

    with right:
        st.subheader("Prediction")
        if st.button("Run prediction", type="primary", width="stretch"):
            payload = {"features": {f: st.session_state[f"f_{f}"] for f in FEATURES}, "model": chosen}
            st.session_state.result = {
                "one": api("POST", "/predict", json=payload),
                "all": api("POST", "/predict/all", json=payload),
            }

        res = st.session_state.result
        if res is None:
            st.caption("Set the features (or load a random patient) and press **Run prediction**.")
        else:
            one = res["one"]
            color = MALIGNANT if one["prediction"] == 1 else BENIGN
            st.markdown(
                f"<div style='padding:14px;border-radius:10px;background:{color}22;border:1px solid {color}'>"
                f"<span style='font-size:0.9em'>{pretty(one['model_type'])} · v{one['version']}"
                f"{' · ⭐ champion' if one['is_champion'] else ''}</span><br>"
                f"<span style='font-size:1.8em;font-weight:700;color:{color}'>{one['label'].upper()}</span></div>",
                unsafe_allow_html=True,
            )
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=one["probability_malignant"] * 100,
                number={"suffix": "%"}, title={"text": "P(malignant)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 50], "color": "#e8f5e9"}, {"range": [50, 100], "color": "#fdecea"}],
                       "threshold": {"line": {"color": "black", "width": 3}, "value": 50}},
            ))
            gauge.update_layout(height=270, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(gauge, width="stretch")

            df = pd.DataFrame(res["all"])
            df["name"] = df.apply(lambda r: f"{'⭐ ' if r.is_champion else ''}{pretty(r.model_type)} (v{r.version})", axis=1)
            fig = px.bar(df.sort_values("probability_malignant"), x="probability_malignant", y="name",
                         orientation="h", color="label",
                         color_discrete_map={"malignant": MALIGNANT, "benign": BENIGN},
                         text=df.sort_values("probability_malignant")["probability_malignant"].map("{:.0%}".format),
                         title="Same patient, every model")
            fig.add_vline(x=0.5, line_dash="dash", annotation_text="threshold 0.5")
            fig.update_layout(height=320, xaxis=dict(range=[0, 1.05], title="P(malignant)", tickformat=".0%"),
                              yaxis_title=None, legend_title=None, margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")
            agree = df["prediction"].nunique() == 1
            st.caption("✅ All models agree." if agree else
                       f"⚠️ Models disagree: {int(df.prediction.sum())} of {len(df)} say malignant.")
            if one["filled_with_median"]:
                st.caption(f"{len(one['filled_with_median'])} features were filled with medians.")

# ================================================================== COMPARE
with tab_cmp:
    ev = get_evaluation()
    E = ev["models"]
    rows = [{"model": pretty(e["model_type"]), "version": e["version"], "champion": "⭐" if e["is_champion"] else "",
             **e["metrics"], "cv_f1_mean": e["cv_f1_mean"], "train_time_sec": e["train_time_sec"]} for e in E]
    lb = pd.DataFrame(rows)

    best = lb.iloc[0]
    k1, k2, k3, k4 = st.columns([1.5, 1, 1.2, 1])
    k1.metric("Champion", f"{best['model']} v{best['version']}")
    k2.metric("Best F1", f"{lb.f1.max():.3f}")
    k3.metric("Best recall (cancers caught)", f"{lb.recall.max():.1%}")
    k4.metric("Test set", f"{ev['n_test']} patients")

    st.subheader("Leaderboard")
    st.dataframe(
        lb.style.highlight_max(subset=METRICS + ["cv_f1_mean"], color="#c8e6c9")
                .format({m: "{:.4f}" for m in METRICS + ["cv_f1_mean"]} | {"train_time_sec": "{:.2f}s"}),
        width="stretch", hide_index=True,
    )

    long = lb.melt(id_vars=["model"], value_vars=METRICS, var_name="metric", value_name="score")
    fig = px.bar(long, x="metric", y="score", color="model", barmode="group",
                 title="Test-set metrics by model (hover for exact values)")
    fig.update_layout(yaxis=dict(range=[max(0, long.score.min() - 0.05), 1.0]), height=420)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    roc = go.Figure()
    pr = go.Figure()
    for e in E:
        name = f"{'⭐ ' if e['is_champion'] else ''}{pretty(e['model_type'])} (AUC {e['metrics']['roc_auc']:.3f})"
        width = 4 if e["is_champion"] else 2
        roc.add_trace(go.Scatter(x=e["roc"]["fpr"], y=e["roc"]["tpr"], mode="lines", name=name, line=dict(width=width)))
        pr.add_trace(go.Scatter(x=e["pr"]["recall"], y=e["pr"]["precision"], mode="lines",
                                name=pretty(e["model_type"]), line=dict(width=width)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dash", color="grey")))
    roc.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
                      height=430, legend=dict(font_size=10))
    pr.update_layout(title="Precision–Recall curves", xaxis_title="Recall", yaxis_title="Precision",
                     height=430, yaxis=dict(range=[0.5, 1.02]), legend=dict(font_size=10))
    c1.plotly_chart(roc, width="stretch")
    c2.plotly_chart(pr, width="stretch")

    st.subheader("Confusion matrices")
    st.caption("Rows = true class, columns = predicted. **Bottom-left (FN) = missed cancers** — the costliest error.")
    ncols = 3
    nrows = -(-len(E) // ncols)
    cm_fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[
        f"{'⭐ ' if e['is_champion'] else ''}{pretty(e['model_type'])}" for e in E],
        horizontal_spacing=0.08, vertical_spacing=0.18)
    labels = ["benign", "malignant"]
    for i, e in enumerate(E):
        cm = e["confusion_matrix"]
        text = [[f"TN {cm[0][0]}", f"FP {cm[0][1]}"], [f"FN {cm[1][0]}", f"TP {cm[1][1]}"]]
        cm_fig.add_trace(go.Heatmap(z=cm, x=labels, y=labels, text=text, texttemplate="%{text}",
                                    colorscale="Blues", showscale=False),
                         row=i // ncols + 1, col=i % ncols + 1)
    cm_fig.update_yaxes(autorange="reversed")
    cm_fig.update_layout(height=300 * nrows, margin=dict(t=40))
    st.plotly_chart(cm_fig, width="stretch")

    st.subheader("Generalization check: cross-validation vs test F1")
    gen = go.Figure()
    gen.add_trace(go.Bar(name="CV F1 (train folds)", x=lb.model, y=lb.cv_f1_mean,
                         error_y=dict(type="data", array=[e["cv_f1_std"] for e in E])))
    gen.add_trace(go.Bar(name="Test F1", x=lb.model, y=lb.f1))
    gen.update_layout(barmode="group", height=380, yaxis=dict(range=[0.8, 1.0]),
                      title="Big gap between bars = overfitting; error bar = CV variance")
    st.plotly_chart(gen, width="stretch")

# ================================================================== REGISTRY
with tab_reg:
    st.subheader("MLflow Model Registry")
    st.markdown(f"Every training run registers its model as a new **version** of "
                f"`breast-cancer-classifier`. The best one by F1 gets the alias **`champion`**, "
                f"which the API serves by default. Full experiment history: [MLflow UI]({MLFLOW_UI_URL}).")
    reg = pd.DataFrame([{
        "version": m["version"], "model": pretty(m["model_type"]),
        "aliases": ", ".join(m["aliases"]) or "—",
        "f1": m["metrics"].get("f1"), "recall": m["metrics"].get("recall"),
        "roc_auc": m["metrics"].get("roc_auc"),
        "params": ", ".join(f"{k}={v}" for k, v in m["params"].items() if k != "model_type"),
        "run_id": m["run_id"],
    } for m in models]).sort_values("version")
    st.dataframe(reg, width="stretch", hide_index=True)

    st.markdown("**Workflow:** retrain (`python -m app.train`) → new versions appear in MLflow "
                "→ press reload so the API picks up the new champion, no restart needed.")
    if st.button("🔄 Reload models from registry"):
        info = api("POST", "/reload")
        st.cache_data.clear()
        st.success(f"Reloaded {info['models_loaded']} models. Champion: {pretty(info['champion'])}")
        st.rerun()
