# forces_tracker_clean.py
import streamlit as st

FORCES_CA = [
    {
        "name": "State litigation vs. OBBB rules",
        "status": "Counter-force",
        "last_updated": "2025-08-10",
        "why_it_matters": "Could pause or narrow federal Medicaid/Marketplace changes affecting coverage and payer mix.",
        "link": "https://oag.ca.gov/news"
    },
    {
        "name": "DHCS Medi-Cal implementation (work reqs, eligibility ops)",
        "status": "Mixed",
        "last_updated": "2025-08-08",
        "why_it_matters": "How counties/plans administer new rules will drive churn risk and uncompensated care exposure.",
        "link": "https://www.dhcs.ca.gov/Pages/Newsroom.aspx"
    },
    {
        "name": "Covered California subsidy 'wrap' for 2026",
        "status": "Counter-force",
        "last_updated": "2025-08-07",
        "why_it_matters": "Potential state subsidies could offset OBBB’s subsidy cliff and stabilize your commercial mix.",
        "link": "https://board.coveredca.com/meetings/2025/"
    },
    {
        "name": "State budget levers & provider tax limits",
        "status": "Mixed",
        "last_updated": "2025-08-02",
        "why_it_matters": "Budget capacity to backfill federal changes is constrained; watch LAO/DHCS updates.",
        "link": "https://lao.ca.gov/"
    },
    {
        "name": "CA protections for gender-affirming care & nondiscrimination",
        "status": "Counter-force",
        "last_updated": "2025-08-01",
        "why_it_matters": "State-regulated plans maintain coverage protections even if federal programs narrow benefits.",
        "link": "https://dmhc.ca.gov/"
    },
]

def _status_style(status: str):
    base = "display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;"
    colors = {
        "Counter-force": "background:#e8f5e9;color:#1b5e20;border:1px solid #c8e6c9;",   # green
        "Amplifier":     "background:#ffebee;color:#b71c1c;border:1px solid #ffcdd2;",   # red
        "Mixed":         "background:#fff8e1;color:#8d6e00;border:1px solid #ffecb3;",   # amber
    }
    return base + colors.get(status, "background:#eceff1;color:#37474f;border:1px solid #cfd8dc;")

def render_forces_tracker(forces=FORCES_CA, title="California Forces Shaping OBBB Impact"):
    st.markdown(f"### {title}")

    colf1, colf2 = st.columns([1,1])
    with colf1:
        status_filter = st.multiselect(
            "Filter by status", options=["Counter-force", "Amplifier", "Mixed"], default=["Counter-force","Amplifier","Mixed"]
        )
    with colf2:
        sort_key = st.selectbox("Sort by", ["Last updated (newest)", "Name (A–Z)"])

    data = [f for f in forces if f["status"] in status_filter]
    if sort_key == "Last updated (newest)":
        data = sorted(data, key=lambda x: x["last_updated"], reverse=True)
    else:
        data = sorted(data, key=lambda x: x["name"].lower())

    for f in data:
        with st.container(border=True):
            c1, c2 = st.columns([0.72, 0.28])
            with c1:
                st.markdown(f"**{f['name']}**")
                st.markdown(
                    f"<span style='{_status_style(f['status'])}'>{f['status']}</span>"
                    f"&nbsp;&nbsp;<span style='font-size:12px;color:#607d8b;'>Last updated: {f['last_updated']}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"🧭 {f['why_it_matters']}")
            with c2:
                st.link_button("Open source", f["link"])

# Example usage
if __name__ == "__main__":
    st.title("🤖 CALIBER360 | OBBB Forces Tracker")
    render_forces_tracker()
