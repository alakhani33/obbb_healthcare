# app_streamlit.py
import os
from datetime import datetime
import streamlit as st

# ---------------------------
# Session state (init)
# ---------------------------
if "question" not in st.session_state:
    st.session_state.question = ""
if "last_logged_q" not in st.session_state:
    st.session_state.last_logged_q = None

# ---------------------------
# Local modules
# ---------------------------
from rag_core import init_chroma, get_embeddings, get_llm, retrieve, format_context
from prompts import SYSTEM_PROMPT, USER_PROMPT
from forces_tracker import FORCES_CA, render_forces_tracker

st.caption(f"Vector backend: {'FAISS' if os.getenv('VECTOR_BACKEND','faiss').lower()=='faiss' else 'Chroma'}")

# ---------------------------
# Google Sheets (no pandas)
# ---------------------------
import gspread
from google.oauth2 import service_account

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_sheet():
    """
    Expects your service account JSON in st.secrets["gdrive"] (as an object, not a string),
    and your Sheet to be shared with that service account email as Editor.
    """
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive"], scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    # Use your Sheet URL or ID
    SHEET_URL = "https://docs.google.com/spreadsheets/d/177G-qeI5NjAPEU4xqLJ_WPZ2xyHdAk-ZaU3QwMmDuQA/edit"
    ss = gc.open_by_url(SHEET_URL)
    return ss.worksheet("Sheet1")  # <-- change to your tab name if different

def log_question_to_sheet(ws, question: str, audience: str):
    ts = datetime.now().isoformat(timespec="seconds")
    ws.append_row(
        [ts, question, audience],
        value_input_option="USER_ENTERED"
    )

# ---------------------------
# Page config + Executive UI
# ---------------------------
st.set_page_config(page_title="CALIBER360 OBBB Executive Advisor", page_icon="🏥", layout="wide")

# Global CSS for exec readability
st.markdown(
    """
    <style>
    /* Global font bump */
    html, body, [class*="css"]  { font-size: 18px; }
    textarea { font-size: 18px !important; }
    .stButton>button { font-size: 18px; padding: 0.5em 1em; }

    /* Caption + labels darker & larger */
    .block-container .stCaption { font-size: 18px !important; color: #000 !important; font-weight: 500 !important; }
    label, .stRadio > label, .stSelectbox > label { font-size: 18px !important; color: #000 !important; font-weight: 600 !important; }
    .stRadio div[role='radiogroup'] label p { font-size: 18px !important; color: #000 !important; font-weight: 500 !important; }

    /* Example chip buttons look tighter */
    .example-chip button { width: 100%; white-space: normal; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helpers
# ---------------------------
def format_forces_for_prompt(forces):
    lines = []
    for f in forces:
        lines.append(
            f"- {f['name']} | Status: {f['status']} | Updated: {f['last_updated']} | "
            f"Why it matters: {f['why_it_matters']} | Link: {f['link']}"
        )
    return "\n".join(lines)

def choose_top_k(question: str) -> int:
    """Dynamic retrieval size. Override with env TOP_K if provided."""
    env_k = os.getenv("TOP_K")
    if env_k and env_k.isdigit():
        return max(2, min(12, int(env_k)))

    ql = (question or "").lower()
    if len(ql.split()) < 10:
        return 4  # short/pointed
    if any(w in ql for w in ["list", "summarize", "summary", "deadlines", "milestones", "penalties", "compare", "impact"]):
        return 10  # broad/summary
    return 6  # default

# ---------------------------
# Header + Blurb
# ---------------------------
st.title("🏥 CALIBER360 OBBB Executive Advisor")
st.markdown(
    """
    <div style="font-size:18px; color:#000; font-weight:500; line-height:1.5; background-color:#f5f9ff; padding:12px; border-left:5px solid #2E86C1;">
    The <strong>One Big Beautiful Bill (OBBB)</strong>, enacted in <strong>July 2025</strong>, reshapes Medicaid, ACA subsidies, and covered services—shifting coverage, uncompensated care risk, and demand across key service lines.
    <br><br>
    But OBBB’s impact isn’t one-way. <strong>Counterforces</strong>—including state litigation, DHCS implementation choices, potential Covered California subsidy wraps, budget constraints, and state-level protections—can <em>moderate</em> or <em>amplify</em> effects. 
    This advisor blends bill text with live counterforces so leaders can see <strong>what changes, for whom, and when</strong>.
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("Answers tailored to Strategy · Operations · Finance — with citations to the bill.")

# ---------------------------
# Audience focus + Examples
# ---------------------------
aud = st.radio("Primary audience focus:", ["All", "Strategy", "Operations", "Finance"], horizontal=True)

# ---------------------------
# Example Questions + Input (session-state based)
# ---------------------------
st.markdown("**Example questions:**")
examples = [
    "When is the big bill expected to be implemented?",
    "What operational changes will be required to comply with the bill?",
    "What are the projected financial penalties for non-compliance?",
    "Which service lines are most affected by the new provisions?",
    "What are the key compliance deadlines and milestones in this bill?",
]

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex_{i}", type="secondary", use_container_width=True):
            st.session_state.question = ex
            st.rerun()  # Immediately refresh to show in text area

# Question input (bound to session state)
q = st.text_area(
    "Ask a question",
    key="question",
    placeholder="e.g., When do the price transparency provisions take effect and what are the penalties for noncompliance?",
    height=120
)

# ---------------------------
# Answer action
# ---------------------------
if st.button("Get Answer", type="primary") and st.session_state.question.strip():
    # 1) Log the question to Google Sheet (with duplicate-write guard)
    q_clean = st.session_state.question.strip()
    try:
        ws = get_sheet()
        if st.session_state.last_logged_q != q_clean:
            log_question_to_sheet(ws, q_clean, aud)
            st.session_state.last_logged_q = q_clean
            # st.toast("Question logged to Google Sheet ✅", icon="✅")
    except Exception as e:
        st.warning(f"Could not log question: {e}")

    # 2) RAG + LLM answer
    with st.spinner("CALIBER360 analyzing bill context and current counterforces…"):
        # Initialize retrieval components
        try:
            _, coll = init_chroma()
            embeddings = get_embeddings()
        except Exception as e:
            st.error(f"Vector DB initialization failed: {e}")
            st.stop()

        # Dynamic, offline-managed top_k
        top_k = choose_top_k(q_clean)

        # Retrieve context safely
        try:
            docs = retrieve(coll, embeddings, q_clean, k=top_k)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            docs = []

        # Prepare contexts
        context = format_context(docs) if docs else "No relevant bill context found."
        forces_context = format_forces_for_prompt(FORCES_CA)

        # Compose messages
        sys_msg = SYSTEM_PROMPT
        user_msg = USER_PROMPT.format(
            question=q_clean,
            context=context,
            forces_context=forces_context,
            audience=aud
        )

        # LLM call
        try:
            llm = get_llm()
            response = llm.invoke([{"role": "system", "content": sys_msg},
                                   {"role": "user", "content": user_msg}])
            st.markdown(
                "<div style='font-size:13px;color:#2E7D32;font-weight:600;'>Counterforces applied where relevant.</div>",
                unsafe_allow_html=True
            )
            st.markdown("### Answer")
            st.write(response.content)
        except Exception as e:
            st.error(f"Model call failed: {e}")

        # Citations
        if docs:
            def make_citations(d):
                meta = d.get("metadata", {})
                title = meta.get("doc_title", "Unknown")
                page = meta.get("page", "?")
                sec = meta.get("section", "?")
                return f"[Source: {title}, p.{page} §{sec}]"

            with st.expander("Citations (retrieved sources)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {make_citations(d)}")

# ---------------------------
# Forces Tracker Panel
# ---------------------------
st.divider()
render_forces_tracker(FORCES_CA, title="California Counterforces that Shape OBBB Impact")


# ---------------------------
# NEXT STEPS: Collaboration Module
# ---------------------------
from gspread.exceptions import WorksheetNotFound

def get_nextsteps_sheet():
    """Open or create a 'NextSteps' worksheet for collaboration leads."""
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive"], scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    SHEET_URL = "https://docs.google.com/spreadsheets/d/177G-qeI5NjAPEU4xqLJ_WPZ2xyHdAk-ZaU3QwMmDuQA/edit"
    ss = gc.open_by_url(SHEET_URL)
    try:
        ws = ss.worksheet("NextSteps")
    except WorksheetNotFound:
        ws = ss.add_worksheet(title="NextSteps", rows=1000, cols=20)
        ws.append_row(
            ["timestamp", "org_name", "contact_name", "email", "role",
             "state", "notes",
             "has_payer_mix", "has_rates", "has_util", "has_costs",
             "has_admin_ops", "has_relief_funds", "has_rcm_metrics"]
        )
    return ws

def render_next_steps():
    st.divider()
    st.markdown(
        """
        <h2 style="color:#B22222; text-align:center; font-weight:800; margin-top:0;">
            🚨 NEXT STEPS: Quantify OBBB Together — Act Now
        </h2>
        <p style="font-size:18px; line-height:1.6;">
        OBBB will rapidly reshape <strong>coverage</strong>, <strong>reimbursement</strong>, and <strong>utilization</strong>.
        Systems that quantify impacts now will protect margins and move first. We invite executives to co-build
        facility-specific scenarios with us using your data.
        </p>
        """,
        unsafe_allow_html=True
    )

    # The big equation (compact)
    st.markdown("#### Our impact model (high-level)")
    st.code(
        "ΔOM_T  =  ΔREV_T  −  ΔCOST_T  +  OFFSETS_T",
        language="text"
    )
    with st.expander("See factor breakdown"):
        st.markdown(
            """
            **Revenue**
            - ΔREV_T = Σ_{p,s} [ (ΔVOL_{p,s} × MARGIN_{p,s,base}) + (VOL_{p,s,post} × ΔRATE_{p,s}) ]

            **Costs**
            - ΔCOST_T = ΔVARCOST_T + ΔFIXCOST_T + ΔADMIN_T + ΔBADDEBT_T + ΔCAPEX/OPEX_T

            **Offsets**
            - OFFSETS_T = FUNDING_T + SAVINGS_T
            """,
            unsafe_allow_html=True,
        )

    # What we need from you (executive checklist)
    st.markdown("#### Data we’ll help you plug in")
    st.markdown(
        """
        - Coverage & payer mix (Medicaid, ACA, uninsured)
        - Reimbursement schedules (SDP/provider-tax exposure, MPFS/QIP deltas)
        - Utilization by service line & payer (pre/post)
        - Cost structures (unit variable cost, FTE/contract labor, overhead)
        - Admin workload (redeterminations, appeals, billing touches)
        - Relief funds / waivers + efficiency/automation initiatives & ROI
        """,
    )

    # Inline form to capture collaboration interest + what they already have
    with st.form("next_steps_form", clear_on_submit=False):
        st.markdown("#### Start collaboration")
        c1, c2 = st.columns([1,1])
        with c1:
            org_name = st.text_input("Organization", placeholder="e.g., Sutter Health – Sacramento")
            contact_name = st.text_input("Your Name", placeholder="e.g., Jane Doe")
            email = st.text_input("Work Email", placeholder="name@org.com")
            role = st.text_input("Role/Title", placeholder="VP Strategy / CFO / COO")
            state = st.text_input("Primary State(s)", placeholder="e.g., CA, NV")
        with c2:
            st.markdown("**Which inputs can you share now?**")
            has_payer_mix   = st.checkbox("Payer mix / covered lives")
            has_rates       = st.checkbox("Reimbursement schedules / rate files")
            has_util        = st.checkbox("Utilization by service line & payer")
            has_costs       = st.checkbox("Unit costs / FTE & contract labor")
            has_admin_ops   = st.checkbox("Admin workload metrics")
            has_relief_fund = st.checkbox("Relief funds / waivers info")
            has_rcm         = st.checkbox("Revenue cycle metrics (charges, denials, collections)")

            notes = st.text_area("Notes / priorities", height=100, placeholder="What decisions or deadlines are you targeting?")

        submit = st.form_submit_button("📩 Start the collaboration")

    if submit:
        if not (org_name and email):
            st.warning("Please provide at least your organization and email.")
        else:
            try:
                ws = get_nextsteps_sheet()
                ts = datetime.now().isoformat(timespec="seconds")
                ws.append_row([
                    ts, org_name, contact_name, email, role, state, notes,
                    "Y" if has_payer_mix else "N",
                    "Y" if has_rates else "N",
                    "Y" if has_util else "N",
                    "Y" if has_costs else "N",
                    "Y" if has_admin_ops else "N",
                    "Y" if has_relief_fund else "N",
                    "Y" if has_rcm else "N",
                ], value_input_option="USER_ENTERED")
                st.success("Thanks! We’ll follow up shortly to kick off your OBBB impact model.")
                st.markdown(
                    """
                    <p style="font-size:16px;">
                    Prefer email? Reach us at <a href="mailto:ali.lakhani@caliber360ai.com">ali.lakhani@caliber360ai.com</a>.
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Could not record your request: {e}")

# ---------------------------
# NEXT STEPS (collaboration)
# ---------------------------
render_next_steps()

# ---------------------------
# Closing CTA (urgent, action-oriented)
# ---------------------------
st.markdown(
    """
    <hr>
    <p style="font-size:16px; color:#000; font-weight:500; line-height:1.5;">
    <strong>OBBB is more than legislation — it’s a seismic shift in how healthcare facilities operate, compete, and stay financially viable.</strong>
    At <strong><a href="https://caliber360ai.com" target="_blank" style="color:#2E86C1; text-decoration:none;">CALIBER360 Healthcare AI</a></strong>, 
    we quantify OBBB’s strategic, operational, and financial impact so you know exactly where you stand,
    what’s at risk, and where to act first. The cost of inaction is high — penalties, missed opportunities, and competitive disadvantages can be immediate and lasting.
    If you’re serious about protecting margins and positioning your organization for success under OBBB,
    <a href="mailto:ali.lakhani@caliber360ai.com" style="color:#2E86C1; font-weight:600;">contact us today</a>.
    </p>
    """,
    unsafe_allow_html=True
)
