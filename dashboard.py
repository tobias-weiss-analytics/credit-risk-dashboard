"""Streamlit dashboard: explainable, cost-sensitive credit scoring (UCI German Credit)."""

from __future__ import annotations

from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from credit_risk.data import (  # noqa: E402
    CODE_LABELS,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    PROTECTED_FEATURES,
    load_german_credit,
    sex_from_status,
)
from credit_risk.explain import explain, make_explainer  # noqa: E402
from credit_risk.model import (  # noqa: E402
    COST_APPROVE_BAD,
    COST_REJECT_GOOD,
    ModelConfig,
    cost_curve,
    decision_metrics,
    train,
)
from credit_risk.segmentation import segment  # noqa: E402

st.set_page_config(page_title="Credit Risk Analytics", page_icon="🏦", layout="wide")

COLORS = {"Approve": "#2e9e5b", "Reject": "#d64541", "Good": "#2e9e5b", "Bad": "#d64541", "New": "#7f8c8d"}
SEGMENT_FEATURES = NUMERIC_FEATURES + list(ORDINAL_FEATURES)


# --------------------------------------------------------------------------- #
# Cached data, model and explainer
# --------------------------------------------------------------------------- #
@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.Series]:
    return load_german_credit()


@st.cache_resource(show_spinner="Training model ...")
def get_model(config: ModelConfig):
    X, y = get_data()
    return train(X, y, config)


@st.cache_resource(show_spinner="Preparing explanations ...")
def get_explainer(config: ModelConfig):
    X, _ = get_data()
    return make_explainer(get_model(config), X)


@st.cache_resource(show_spinner="Computing SHAP values ...")
def get_global_shap(config: ModelConfig):
    X, _ = get_data()
    sample = X.sample(300, random_state=42)
    return explain(get_model(config), get_explainer(config), sample, readable=False)


@st.cache_resource
def get_segmentation(features: tuple[str, ...], k: int):
    X, _ = get_data()
    return segment(X, list(features), k)


def init_state() -> None:
    defaults = {
        "config": ModelConfig(),
        "threshold": None,
        "threshold_config": None,
        "segment_features": ("Credit_Amount", "Duration", "Installment_Rate"),
        "k": 3,
        "segment_names": {},
        "saved_cases": pd.DataFrame(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()
config: ModelConfig = st.session_state["config"]
X, y = get_data()
model = get_model(config)
explainer = get_explainer(config)

# The threshold defaults to the cost-optimal value and resets after retraining.
if st.session_state["threshold_config"] != config:
    st.session_state["threshold"] = model.threshold
    st.session_state["threshold_config"] = config
threshold: float = st.session_state["threshold"]

seg = get_segmentation(tuple(st.session_state["segment_features"]), st.session_state["k"])
seg_key = (tuple(seg.features), st.session_state["k"])
segment_names = {i: st.session_state["segment_names"].get((seg_key, i), name) for i, name in seg.names.items()}


def decision(score) -> np.ndarray:
    return np.where(np.asarray(score) >= threshold, "Approve", "Reject")


def build_portfolio() -> pd.DataFrame:
    """Historical applicants with out-of-fold scores, plus cases saved in this session."""
    df = X.copy()
    df["Actual"] = y.map({1: "Good", 0: "Bad"})
    df["Score"] = model.oof_proba
    df["Segment"] = [segment_names[i] for i in seg.labels]
    saved = st.session_state["saved_cases"]
    if not saved.empty:
        new = saved[FEATURE_NAMES].copy()
        new.index = pd.RangeIndex(len(X), len(X) + len(new))
        new["Actual"] = "New"
        new["Score"] = model.predict_proba_good(new)
        new["Segment"] = [segment_names[i] for i in seg.assign(new)]
        df = pd.concat([df, new])
    df["Decision"] = decision(df["Score"])
    return df


def waterfall(explanation, max_display: int = 8, height: float = 4.0) -> None:
    fig = plt.figure(figsize=(9, height))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    st.pyplot(fig, clear_figure=True)
    plt.close("all")


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.1%}"


portfolio = build_portfolio()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🏦 Credit Risk Analytics")
page = st.sidebar.radio("Navigation", ["Portfolio", "Loan Application", "Model & Fairness"])
st.sidebar.caption(
    f"Decision threshold: **{threshold:.2f}** on P(good credit)  \n"
    f"Cost matrix: approving a bad applicant = {COST_APPROVE_BAD}, rejecting a good one = {COST_REJECT_GOOD}"
)
st.sidebar.divider()

# --------------------------------------------------------------------------- #
# Page: Portfolio
# --------------------------------------------------------------------------- #
if page == "Portfolio":
    st.title("Portfolio Overview")
    st.caption("Scores are out-of-fold predictions: no applicant is scored by a model that was trained on them.")

    st.sidebar.header("Filter")
    purpose = st.sidebar.selectbox("Purpose", ["All"] + sorted(portfolio["Purpose"].unique()))
    job = st.sidebar.selectbox("Job", ["All"] + sorted(portfolio["Job"].unique()))
    df = portfolio
    if purpose != "All":
        df = df[df["Purpose"] == purpose]
    if job != "All":
        df = df[df["Job"] == job]
    if df.empty:
        st.info("No applications match the filter.")
        st.stop()

    hist = df[df["Actual"] != "New"]
    approved = hist[hist["Decision"] == "Approve"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Applications", f"{len(df):,}")
    k2.metric("Credit volume", f"{df['Credit_Amount'].sum():,.0f} DM")
    k3.metric("Approval rate", pct((df["Decision"] == "Approve").mean()))
    k4.metric("Default rate among approved", pct((approved["Actual"] == "Bad").mean()) if len(approved) else "–",
              help=f"Historical default rate of approved applicants; overall default rate is {pct((hist['Actual'] == 'Bad').mean())}.")

    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Drivers (SHAP)", "Repayment history", "Segments", "Customer deep dive"])

    with t1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Decisions")
            fig = px.pie(df, names="Decision", color="Decision", hole=0.45, color_discrete_map=COLORS)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Approval rate by purpose")
            grp = df.groupby("Purpose").agg(Applications=("Decision", "size"),
                                            Approval_rate=("Decision", lambda s: (s == "Approve").mean()))
            grp = grp.sort_values("Approval_rate").reset_index()
            fig = px.bar(grp, x="Approval_rate", y="Purpose", orientation="h", text_auto=".0%",
                         hover_data=["Applications"], labels={"Approval_rate": "Approval rate"})
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig, width="stretch")

    with t2:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("What drives the score")
            st.caption("SHAP values on P(good credit), 300 applicants. Right = pushes towards approval, "
                       "left = towards rejection. Colour = feature value (grey = outside the ordinal scale).")
            fig = plt.figure(figsize=(9, 6))
            shap.plots.beeswarm(get_global_shap(config), max_display=12, show=False)
            st.pyplot(fig, clear_figure=True)
            plt.close("all")
        with c2:
            st.subheader("Approval rate by category")
            feat = st.selectbox("Feature", [f for f in FEATURE_NAMES if f not in NUMERIC_FEATURES])
            ct = df.groupby(feat)["Decision"].apply(lambda s: (s == "Approve").mean()).sort_values().reset_index()
            fig = px.bar(ct, x="Decision", y=feat, orientation="h", text_auto=".0%", labels={"Decision": "Approval rate"})
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig, width="stretch")
            if feat in PROTECTED_FEATURES and config.exclude_protected:
                st.caption("This attribute is not used by the model; see *Model & Fairness → Fairness*.")

    with t3:
        st.subheader("Historical repayment rate")
        by = st.selectbox("Group by", ["Purpose", "Credit_History", "Checking_Status", "Savings_Account", "Housing", "Job"])
        rep = hist.groupby(by)["Actual"].agg(Good_rate=lambda s: (s == "Good").mean(), Applications="size")
        rep = rep.sort_values("Good_rate").reset_index()
        fig = px.bar(rep, x="Good_rate", y=by, orientation="h", text_auto=".0%", color="Good_rate",
                     color_continuous_scale="RdYlGn", range_color=[0.3, 1], hover_data=["Applications"],
                     labels={"Good_rate": "Repaid"})
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    with t4:
        st.subheader("Customer segments")
        st.caption(f"k-means (k = {st.session_state['k']}) on: {', '.join(seg.features)}. "
                   "Configure under *Model & Fairness → Segmentation*.")
        hist_seg = portfolio[portfolio["Actual"] != "New"].copy()
        hist_seg["PC1"], hist_seg["PC2"] = seg.coords[:, 0], seg.coords[:, 1]
        hist_seg = hist_seg.loc[hist_seg.index.isin(df.index)]
        fig = px.scatter(hist_seg, x="PC1", y="PC2", color="Segment", symbol="Decision", opacity=0.75,
                         hover_data=["Credit_Amount", "Duration", "Purpose", "Actual"])
        st.plotly_chart(fig, width="stretch")
        summary = df.groupby("Segment").agg(
            Applications=("Score", "size"),
            Avg_amount=("Credit_Amount", "mean"),
            Avg_duration=("Duration", "mean"),
            Avg_score=("Score", "mean"),
            Approval_rate=("Decision", lambda s: (s == "Approve").mean()),
        )
        good = hist.groupby("Segment")["Actual"].apply(lambda s: (s == "Good").mean()).rename("Historical_repayment")
        summary = summary.join(good)
        st.dataframe(summary.style.format({"Avg_amount": "{:,.0f} DM", "Avg_duration": "{:.1f} months",
                                           "Avg_score": "{:.1%}", "Approval_rate": "{:.1%}",
                                           "Historical_repayment": "{:.1%}"}), width="stretch")

    with t5:
        st.subheader("Customer deep dive")
        cid = st.selectbox("Customer ID", df.index.tolist()[::-1])
        row = df.loc[cid]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Decision", row["Decision"])
        k2.metric("P(good credit)", pct(row["Score"]))
        k3.metric("Actual outcome", row["Actual"])
        k4.metric("Segment", row["Segment"])
        st.caption("The waterfall uses the final model; the score above is out-of-fold and can differ slightly.")
        waterfall(explain(model, explainer, df.loc[[cid], FEATURE_NAMES])[0], max_display=9)
        with st.expander("Application data"):
            st.dataframe(row[FEATURE_NAMES].astype(str).rename("Value"), width="stretch")

# --------------------------------------------------------------------------- #
# Page: Loan application
# --------------------------------------------------------------------------- #
elif page == "Loan Application":
    st.title("Loan Application")
    sb = st.sidebar

    def pick(label: str, feature: str, default: str | None = None) -> str:
        options = list(CODE_LABELS[feature].values())
        return sb.selectbox(label, options, index=options.index(default) if default else 0)

    sb.header("Loan")
    amount = sb.number_input("Credit amount (DM)", 250, 20000, 4000, step=250)
    duration = sb.slider("Duration (months)", 4, 72, 24)
    rate = sb.slider("Instalment rate (% of disposable income, 1 = low, 4 = high)", 1, 4, 3)
    purpose = pick("Purpose", "Purpose", "Car (used)")
    sb.header("Financial situation")
    checking = pick("Checking account", "Checking_Status", "0-200 DM")
    savings = pick("Savings", "Savings_Account", "100-500 DM")
    history = pick("Credit history", "Credit_History", "Existing paid back duly")
    n_credits = sb.number_input("Existing credits at this bank", 1, 4, 1)
    other_plans = pick("Other instalment plans", "Other_Plans", "None")
    debtors = pick("Other debtors / guarantors", "Debtors", "None")
    prop = pick("Most valuable property", "Property", "Car / other")
    sb.header("Applicant")
    age = sb.number_input("Age (years)", 19, 75, 35)
    employment = pick("Employed since", "Employment_Since", "1-4 years")
    job = pick("Job", "Job", "Skilled employee")
    housing = pick("Housing", "Housing", "Own")
    residence = sb.slider("Present residence since (years, capped at 4)", 1, 4, 2)
    dependents = sb.number_input("Dependents", 1, 2, 1)
    phone = pick("Telephone registered", "Phone", "Yes")
    with sb.expander("Protected attributes" + (" (not used by the model)" if config.exclude_protected else "")):
        sex_status = pick("Personal status & sex", "Sex_Status", "Male: single")
        foreign = pick("Foreign worker", "Foreign_Worker", "No")

    application = pd.DataFrame([{
        "Checking_Status": checking, "Duration": duration, "Credit_History": history, "Purpose": purpose,
        "Credit_Amount": amount, "Savings_Account": savings, "Employment_Since": employment,
        "Installment_Rate": rate, "Sex_Status": sex_status, "Debtors": debtors, "Residence_Since": residence,
        "Property": prop, "Age": age, "Other_Plans": other_plans, "Housing": housing, "Num_Credits": n_credits,
        "Job": job, "Num_Dependents": dependents, "Phone": phone, "Foreign_Worker": foreign,
    }])[FEATURE_NAMES]

    score = float(model.predict_proba_good(application)[0])
    seg_name = segment_names[int(seg.assign(application)[0])]

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Decision")
        if score >= threshold:
            st.success("## ✅ Approve")
        else:
            st.error("## 🛑 Reject")
        st.metric("P(good credit)", pct(score), delta=f"{score - threshold:+.2f} vs. threshold {threshold:.2f}",
                  delta_color="normal")
        st.info(f"Segment: **{seg_name}**")
        if st.button("💾 Add to portfolio"):
            st.session_state["saved_cases"] = pd.concat([st.session_state["saved_cases"], application], ignore_index=True)
            st.toast("Application added to the portfolio.", icon="✅")
    with right:
        st.subheader("Why this decision?")
        st.caption("Contribution of each feature to P(good credit), starting from the average applicant.")
        waterfall(explain(model, explainer, application)[0], max_display=10, height=4.5)

    saved = st.session_state["saved_cases"]
    if not saved.empty:
        st.caption(f"{len(saved)} application(s) added in this session; they appear in the portfolio as 'New'.")

# --------------------------------------------------------------------------- #
# Page: Model & fairness
# --------------------------------------------------------------------------- #
else:
    st.title("Model & Fairness")
    t1, t2, t3, t4 = st.tabs(["Threshold & cost", "Fairness", "Segmentation", "Training"])

    with t1:
        st.caption(f"Evaluated on the hold-out test set ({len(model.y_test)} applicants). The cost-optimal "
                   "threshold was chosen on cross-validated predictions of the training set only.")
        c1, c2 = st.columns([3, 1])
        with c1:
            new_t = st.slider("Decision threshold on P(good credit)", 0.0, 1.0, float(threshold), 0.01,
                              help="Approve if P(good credit) ≥ threshold. Higher = stricter.")
        with c2:
            st.metric("Cost-optimal threshold", f"{model.threshold:.2f}")
            if st.button("Reset to optimum"):
                new_t = model.threshold
        if new_t != threshold:
            st.session_state["threshold"] = new_t
            st.rerun()

        m = decision_metrics(model.y_test, model.proba_test, threshold)
        base = model.metrics["cost_approve_all"]
        a, b, c, d, e = st.columns(5)
        a.metric("ROC-AUC", f"{model.metrics['roc_auc']:.3f}")
        b.metric("Total cost", m["cost"], delta=f"{m['cost'] - base:+d} vs. approve all", delta_color="inverse",
                 help=f"Approving every applicant would cost {base}.")
        c.metric("Approval rate", pct(m["approval_rate"]))
        d.metric("Bad rate among approved", pct(m["bad_rate_among_approved"]))
        e.metric("Accuracy", pct(m["accuracy"]))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### Cost by threshold")
            curve = cost_curve(model.y_test, model.proba_test)
            fig = px.line(curve, x="threshold", y="cost", labels={"threshold": "Threshold", "cost": "Total cost"})
            fig.add_vline(x=threshold, line_dash="dash", line_color="#d64541", annotation_text="current")
            fig.add_vline(x=model.threshold, line_dash="dot", line_color="#2e9e5b", annotation_text="optimum",
                          annotation_position="bottom right")
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.markdown("##### Confusion matrix")
            z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
            fig = px.imshow(z, x=["Rejected", "Approved"], y=["Bad (actual)", "Good (actual)"], text_auto=True,
                            color_continuous_scale="Blues", range_color=[0, len(model.y_test)])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Approved bad applicants cost {COST_APPROVE_BAD} each, rejected good ones {COST_REJECT_GOOD}.")
        with col3:
            st.markdown("##### ROC curve")
            fpr, tpr, _ = roc_curve(model.y_test, model.proba_test)
            fig = px.area(x=fpr, y=tpr, labels={"x": "Approved bad / all bad", "y": "Approved good / all good"})
            fig.add_shape(type="line", line={"dash": "dash"}, x0=0, x1=1, y0=0, y1=1)
            cur_fpr = m["fp"] / (m["fp"] + m["tn"]) if m["fp"] + m["tn"] else 0
            fig.add_trace(go.Scatter(x=[cur_fpr], y=[m["recall_good"]], mode="markers",
                                     marker={"color": "#d64541", "size": 12}, name="current"))
            st.plotly_chart(fig, width="stretch")

    with t2:
        st.subheader("Approval rates across groups")
        status = "excluded from" if config.exclude_protected else "**included in**"
        st.caption(f"Sex/personal status and foreign-worker status are {status} the model. Age is used, as is "
                   "common in credit scoring, and therefore monitored here. Ratio = approval rate of the group "
                   "divided by the highest group's rate; values below 0.8 fail the four-fifths rule of thumb.")
        hist = portfolio[portfolio["Actual"] != "New"]
        groups = {
            "Sex": sex_from_status(hist["Sex_Status"]),
            "Foreign worker": hist["Foreign_Worker"],
            "Age group": pd.cut(hist["Age"], [0, 25, 40, 120], labels=["≤ 25", "26-40", "> 40"]),
        }
        cols = st.columns(len(groups))
        for col, (name, g) in zip(cols, groups.items(), strict=True):
            tab = pd.DataFrame({"group": g, "approve": hist["Decision"] == "Approve", "good": hist["Actual"] == "Good"})
            res = tab.groupby("group", observed=True).agg(n=("approve", "size"), Approval_rate=("approve", "mean"),
                                                          Repaid=("good", "mean"))
            res["Ratio"] = res["Approval_rate"] / res["Approval_rate"].max()
            with col:
                st.markdown(f"##### {name}")
                fig = px.bar(res.reset_index(), x="group", y="Approval_rate", text_auto=".0%",
                             color=res["Ratio"].lt(0.8).map({True: "below 0.8", False: "ok"}).values,
                             color_discrete_map={"ok": "#2e9e5b", "below 0.8": "#e67e22"},
                             labels={"group": "", "Approval_rate": "Approval rate", "color": "Ratio"})
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                st.plotly_chart(fig, width="stretch")
                st.dataframe(res.style.format({"Approval_rate": "{:.1%}", "Repaid": "{:.1%}",
                                               "Ratio": "{:.2f}"}), width="stretch")
        st.caption("Differences in approval rates partly mirror differences in historical repayment. "
                   "Excluding an attribute does not remove its proxies (e.g. age and personal status are correlated).")

    with t3:
        st.subheader("Segmentation")
        feats = st.multiselect("Features", SEGMENT_FEATURES, default=list(st.session_state["segment_features"]))
        k = st.slider("Number of segments (k)", 2, 6, st.session_state["k"])
        if feats and (tuple(feats) != tuple(st.session_state["segment_features"]) or k != st.session_state["k"]):
            st.session_state["segment_features"], st.session_state["k"] = tuple(feats), k
            st.rerun()
        if not feats:
            st.warning("Select at least one feature.")
        st.markdown("##### Segment names")
        st.caption("Generated from each segment's most distinctive features; edit to rename.")
        name_cols = st.columns(len(seg.names))
        for i, col in enumerate(name_cols):
            new_name = col.text_input(f"Segment {i}", segment_names[i], key=f"seg_{seg_key}_{i}")
            if new_name != segment_names[i]:
                st.session_state["segment_names"][(seg_key, i)] = new_name
                st.rerun()

    with t4:
        st.subheader("Training")
        with st.form("training"):
            c1, c2 = st.columns(2)
            n_est = c1.slider("Trees (n_estimators)", 100, 600, config.n_estimators, 50)
            depth = c1.slider("Max. tree depth", 2, 8, config.max_depth)
            lr = c1.select_slider("Learning rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], config.learning_rate)
            test_size = c2.slider("Test set share", 0.1, 0.4, config.test_size, 0.05)
            excl = c2.toggle("Exclude protected attributes", config.exclude_protected)
            mono = c2.toggle("Monotone constraints on ordinal features", config.monotone)
            if st.form_submit_button("Retrain model", type="primary"):
                st.session_state["config"] = replace(config, n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                                     test_size=test_size, exclude_protected=excl, monotone=mono)
                st.rerun()
        if config != ModelConfig() and st.button("Restore defaults"):
            st.session_state["config"] = ModelConfig()
            st.rerun()
        st.markdown("##### Current configuration")
        st.json(asdict(config))
