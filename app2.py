from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="L1 Copilot | RideOps", page_icon="🧭", layout="wide")

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "rideops_support_tickets.csv"
AVOIDABLE = [
    "Config issue - returned to L1",
    "Insufficient info - returned to L1",
    "Duplicate of known issue",
    "Not a bug - how-to",
]
L1_OUTCOMES = AVOIDABLE[:2] + ["Not a bug - how-to"]

# Proposed investigation prompts. These are workflow suggestions, not fields
# proven mandatory by the historical CSV.
PLAYBOOKS = {
    "Roster / booking not syncing": [
        "Confirm client, affected employee and trip/roster ID",
        "Confirm booking date and approximate time of the sync failure",
        "Check whether the issue affects one employee or multiple employees",
        "Check the approved client booking cutoff and roster configuration",
        "Capture screenshot or export showing the mismatch",
        "Search for an existing incident with the same symptom",
    ],
    "Live tracking / GPS incorrect": [
        "Capture trip ID, vehicle/driver ID and time of observation",
        "Confirm whether the vehicle is moving and the app is online",
        "Check whether the issue affects one trip or multiple trips",
        "Capture the map view and last location update time",
        "Check for an existing incident affecting the same client or route",
    ],
    "Report export failing": [
        "Capture report type, date range and client ID",
        "Record the time and exact step where export fails",
        "Check whether the issue occurs for another date range or report",
        "Capture error message or screenshot",
        "Search for a known report-export incident",
    ],
    "OTP / boarding issue": [
        "Capture trip ID, employee ID and time of attempted boarding",
        "Record the exact OTP or boarding error shown",
        "Confirm whether one employee or multiple employees are affected",
        "Check connectivity and the approved boarding workflow",
        "Attach screenshot or other available evidence",
    ],
    "Login / access issue": [
        "Capture user ID, client and affected module",
        "Record the exact error and time it occurred",
        "Confirm the user's role and expected access level",
        "Check whether another authorised user can access the module",
        "Attach screenshot of the error",
    ],
}
DEFAULT_PLAYBOOK = [
    "Confirm client, relevant record ID and time of occurrence",
    "Record the exact steps that reproduce the issue",
    "Confirm scope: one user/record or multiple users/records",
    "Capture the exact error message and supporting evidence",
    "Search for a similar existing ticket or known incident",
]


def norm(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(upload=None):
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    required = {"ticket_id", "created_at", "category", "description", "escalated_to_eng", "eng_outcome", "client_id", "status", "eng_handoffs", "reopen_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    for col in ["created_at", "first_response_at", "first_closed_at", "final_closed_at"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["category", "description", "eng_outcome", "status", "client_name", "client_id", "priority"]:
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    df["category"] = df["category"].replace("", "Uncategorised")
    df["description"] = df["description"].fillna("")
    df["esc"] = df["escalated_to_eng"].astype(str).str.lower().str.strip().isin(["yes", "true", "1"])
    df["eng_handoffs"] = pd.to_numeric(df["eng_handoffs"], errors="coerce").fillna(0)
    df["reopen_count"] = pd.to_numeric(df["reopen_count"], errors="coerce").fillna(0)
    df["csat_score"] = pd.to_numeric(df.get("csat_score", np.nan), errors="coerce")
    df["norm_description"] = df["description"].map(norm)
    df["search_text"] = (df["category"] + " " + df["description"]).str.strip()
    df["final_hours"] = (df["final_closed_at"] - df["created_at"]).dt.total_seconds() / 3600 if "final_closed_at" in df else np.nan
    df["avoidable_outcome"] = df["esc"] & df["eng_outcome"].isin(AVOIDABLE)
    return df


@st.cache_data(show_spinner=False)
def cached_load(upload_bytes=None):
    from io import BytesIO
    return load_data(BytesIO(upload_bytes) if upload_bytes else None)


@st.cache_resource(show_spinner=False)
def build_index(texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(list(texts))
    return vec, mat


def similar_tickets(df, category, query, k=10, exclude_id=None):
    pool = df[df.category.eq(category)].copy()
    if exclude_id is not None:
        pool = pool[pool.ticket_id.ne(exclude_id)]
    if pool.empty or not norm(query):
        pool["similarity"] = []
        return pool
    corpus = tuple(pool.search_text.tolist())
    try:
        vec, mat = build_index(corpus)
        q = vec.transform([category + " " + query])
        scores = cosine_similarity(q, mat).ravel()
    except ValueError:
        pool["similarity"] = 0.0
        return pool.iloc[0:0]
    pool["similarity"] = scores
    return pool[pool.similarity > 0].sort_values("similarity", ascending=False).head(k)


def outcome_summary(matches):
    evidence = matches[matches.esc & matches.eng_outcome.ne("")]
    if evidence.empty:
        return evidence, pd.Series(dtype=int)
    return evidence, evidence.eng_outcome.value_counts()


def recommendation(evidence):
    n = len(evidence)
    if n < 3:
        return "Insufficient historical evidence", "Fewer than three similar escalated tickets have recorded outcomes. Use the checklist and apply agent judgement."
    shares = evidence.eng_outcome.value_counts(normalize=True)
    avoid_share = evidence.eng_outcome.isin(AVOIDABLE).mean()
    if avoid_share >= 0.5:
        if shares.get("Duplicate of known issue", 0) > evidence.eng_outcome.isin(L1_OUTCOMES).mean():
            return "Search for a known incident first", f"{avoid_share:.0%} of similar escalations ended in one of the selected non-fix outcomes."
        return "Complete L1 checks before escalation", f"{avoid_share:.0%} of similar escalations ended in one of the selected non-fix outcomes."
    fixed_share = evidence.eng_outcome.eq("Fixed").mean()
    if fixed_share >= 0.5:
        return "Escalation is supported by history", f"{fixed_share:.0%} of similar escalations were marked Fixed. Verify the issue and attach evidence before handoff."
    return "Investigate, then use judgement", f"Historical outcomes are mixed; {avoid_share:.0%} were in the selected non-fix outcome group."


def fmt_hours(x):
    return "n/a" if pd.isna(x) else f"{x:.1f} h"


def make_handoff(category, description, client, record_id, occurred, checks, extra):
    lines = ["ENGINEERING HANDOFF", f"Category: {category}", f"Client: {client or 'Not provided'}", f"Trip / roster / record ID: {record_id or 'Not provided'}", f"Time of occurrence: {occurred or 'Not provided'}", "", "Issue description:", description or "Not provided", "", "L1 checks completed:"]
    lines += [f"- {x}" for x in checks] if checks else ["- None recorded"]
    lines += ["", "Evidence / error details:", extra or "Not provided"]
    return "\n".join(lines)


# Sidebar: data source and navigation
st.sidebar.title("L1 Copilot")
st.sidebar.caption("RideOps · Internal support workspace")
upload = st.sidebar.file_uploader("Replace historical dataset (CSV)", type=["csv"])
try:
    df = cached_load(upload.getvalue() if upload else None)
except Exception as exc:
    st.error(str(exc))
    st.stop()

page = st.sidebar.radio("Workspace", ["Ticket investigation", "Incident radar", "Operations overview"])
st.sidebar.divider()
st.sidebar.caption(f"Loaded {len(df):,} tickets · Historical snapshot through {df.created_at.max().date() if df.created_at.notna().any() else 'unknown'}")
st.sidebar.caption("Historical evidence only. No live system checks or ticket-system integration.")

st.title("L1 Copilot")
st.caption("Investigate tickets, improve handoffs and identify recurring incidents using historical support data.")

if page == "Ticket investigation":
    st.subheader("Ticket investigation workspace")
    left, right = st.columns([1.05, 1])
    with left:
        category = st.selectbox("Issue category", sorted(df.category.unique()))
        description = st.text_area("Describe the issue", height=120, placeholder="Example: Cancelled bookings are still appearing in tomorrow's roster")
        client = st.text_input("Client name / ID", placeholder="e.g. Harbor Health / CL-177")
        record_id = st.text_input("Trip, roster or relevant record ID")
        occurred = st.text_input("When did the issue occur?", placeholder="Date and approximate time")
        extra = st.text_area("Error message / evidence notes", height=75)
        find = st.button("Investigate ticket", type="primary", use_container_width=True)
    with right:
        st.markdown("#### Investigation checklist")
        checklist = PLAYBOOKS.get(category, DEFAULT_PLAYBOOK)
        checked = []
        for i, item in enumerate(checklist):
            if st.checkbox(item, key=f"check_{category}_{i}"):
                checked.append(item)
        st.caption("These are proposed prompts. Validate them with support agents before operational use.")

    if find:
        if not description.strip():
            st.warning("Enter an issue description before investigating.")
        else:
            matches = similar_tickets(df, category, description, k=12)
            evidence, counts = outcome_summary(matches)
            rec, why = recommendation(evidence)
            st.session_state["investigation"] = {
                "category": category, "description": description, "client": client,
                "record_id": record_id, "occurred": occurred, "extra": extra,
                "checked": checked, "matches": matches, "evidence": evidence,
                "recommendation": rec, "reason": why,
            }
    result = st.session_state.get("investigation")
    if result:
        st.divider()
        st.markdown("### Investigation results")
        a, b, c = st.columns(3)
        a.metric("Similar historical tickets", len(result["matches"]))
        b.metric("Similar escalations with outcomes", len(result["evidence"]))
        med = result["evidence"].final_hours.median() if len(result["evidence"]) and "final_hours" in result["evidence"] else np.nan
        c.metric("Median final resolution", fmt_hours(med))
        st.info(f"**Suggested next step: {result['recommendation']}**\n\n{result['reason']}")
        if not result["evidence"].empty:
            st.markdown("#### Historical outcome mix among similar escalations")
            mix = result["evidence"].eng_outcome.value_counts().rename_axis("Outcome").reset_index(name="Tickets")
            st.bar_chart(mix.set_index("Outcome"))
        else:
            st.warning("No similar escalated tickets with recorded outcomes were found. Do not infer a recommendation from this result.")
        st.markdown("#### Similar historical tickets")
        show_cols = [c for c in ["ticket_id", "client_name", "description", "eng_outcome", "eng_handoffs", "final_hours", "similarity"] if c in result["matches"].columns]
        display = result["matches"][show_cols].copy()
        if "similarity" in display:
            display["similarity"] = (display["similarity"] * 100).round(1).astype(str) + "%"
        if "final_hours" in display:
            display["final_hours"] = display["final_hours"].round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown("#### Engineering handoff draft")
        handoff = make_handoff(result["category"], result["description"], result["client"], result["record_id"], result["occurred"], result["checked"], result["extra"])
        st.text_area("Copy this into your ticketing system", handoff, height=260)
        st.download_button("Download handoff as .txt", handoff, file_name="engineering_handoff.txt", mime="text/plain")

elif page == "Incident radar":
    st.subheader("Recurring incident radar")
    st.write("Find repeated issue descriptions and retrieve related reports within a category. Similarity is text-based, not proof that tickets share the same root cause.")
    cats = ["All categories"] + sorted(df.category.unique().tolist())
    chosen = st.selectbox("Filter category", cats)
    min_count = st.slider("Minimum reports in an exact-description group", 2, 15, 3)
    subset = df if chosen == "All categories" else df[df.category.eq(chosen)]
    grouped = (subset[subset.norm_description.ne("")]
               .groupby(["category", "norm_description"], dropna=False)
               .agg(ticket_count=("ticket_id", "count"), first_reported=("created_at", "min"), last_reported=("created_at", "max"), clients=("client_id", "nunique"), sample_description=("description", "first"))
               .reset_index())
    grouped = grouped[grouped.ticket_count >= min_count].sort_values(["ticket_count", "clients"], ascending=False)
    if grouped.empty:
        st.info("No exact-description groups meet this threshold.")
    else:
        st.metric("Recurring issue patterns", len(grouped))
        st.dataframe(grouped[["category", "sample_description", "ticket_count", "clients", "first_reported", "last_reported"]].rename(columns={"sample_description": "Issue description", "ticket_count": "Tickets", "clients": "Unique clients", "first_reported": "First reported", "last_reported": "Last reported"}), use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("#### Search for related reports")
    q1, q2 = st.columns([1, 2])
    with q1:
        radar_cat = st.selectbox("Category for similarity search", sorted(df.category.unique()), key="radar_cat")
    with q2:
        radar_query = st.text_input("Describe the recurring symptom", placeholder="e.g. Cancelled bookings still visible in roster")
    if st.button("Find related tickets", key="radar_search"):
        related = similar_tickets(df, radar_cat, radar_query, k=30)
        if related.empty:
            st.warning("No text-similar tickets found in this category.")
        else:
            clients = related.client_name.nunique() if "client_name" in related else related.client_id.nunique()
            st.success(f"Found {len(related)} related tickets across {clients} distinct clients. Review descriptions to confirm whether they represent one incident.")
            st.dataframe(related[[c for c in ["ticket_id", "created_at", "client_name", "description", "eng_outcome", "status", "similarity"] if c in related.columns]], use_container_width=True, hide_index=True)

else:
    st.subheader("Support operations overview")
    total = len(df)
    escalated = df[df.esc]
    open_now = df.status.str.lower().eq("open").sum()
    target = escalated.eng_outcome.isin(AVOIDABLE).sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets", f"{total:,}")
    k2.metric("Escalated to Engineering", f"{len(escalated):,}", f"{len(escalated)/total:.1%} of tickets" if total else "")
    k3.metric("Open at snapshot", f"{open_now:,}")
    k4.metric("Selected non-fix outcomes", f"{target:,}", f"{target/len(escalated):.1%} of escalations" if len(escalated) else "")
    st.caption("Selected non-fix outcomes include configuration return, insufficient information return, duplicate and how-to. This is an outcome-based opportunity measure, not a verified count of physical returns.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Ticket volume by category")
        st.bar_chart(df.category.value_counts().head(10))
    with c2:
        st.markdown("#### Engineering outcomes")
        outcomes = escalated.eng_outcome.replace("", "Not recorded").value_counts()
        st.bar_chart(outcomes)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Weekly ticket inflow")
        weekly = df.dropna(subset=["created_at"]).set_index("created_at").resample("W").size()
        st.line_chart(weekly)
    with c4:
        st.markdown("#### Handoffs per escalated ticket")
        if len(escalated):
            st.bar_chart(escalated.eng_handoffs.value_counts().sort_index())
    st.markdown("#### Resolution time: escalated vs non-escalated")
    comparison = df.assign(Group=np.where(df.esc, "Escalated", "Not escalated")).groupby("Group").final_hours.median()
    st.dataframe(comparison.rename("Median final resolution (hours)").to_frame(), use_container_width=True)
    st.warning("The CSV is a 30 September snapshot. It does not contain historical backlog snapshots, current ticket ownership, actual troubleshooting steps or live configuration state. Do not infer historical backlog growth or current Engineering ownership from the ever-escalated flag.")

st.sidebar.divider()
st.sidebar.caption("Prototype · Advisory decision support · No ticket actions are executed")
