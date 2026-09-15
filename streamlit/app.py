"""
Insurance Claims Intelligence & Fraud Detection Platform.

A production-inspired enterprise Streamlit application combining Snowflake analytics,
machine learning inference, and MLOps experiment tracking.
"""

import sys
from pathlib import Path

# Add project root and streamlit folder to sys.path
STREAMLIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMLIT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STREAMLIT_DIR))

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.config import ARTIFACTS_DIR, CHAMPION_ALIAS, DEFAULT_OPERATING_THRESHOLD
from ml.predict import predict_claim_risk
from ml.utils import get_sample_claims
from utils.formatting import (
    format_currency,
    format_number,
    format_percentage,
    get_risk_badge_html,
)
from utils.model import get_cached_champion_model
from utils.snowflake import (
    get_claims_by_incident_type,
    get_claims_by_severity,
    get_claims_by_state,
    get_fraud_analysis,
    get_kpis,
    get_monthly_claims,
    load_analytical_base_data,
)

# Page configuration
st.set_page_config(
    page_title="Insurance Claims Intelligence & Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-card-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar():
    """Renders the navigation sidebar and architecture metadata."""
    st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.sidebar.title("Claims Intelligence")
    st.sidebar.caption("Portfolio Platform | Verisk DS/ML/DE")

    page = st.sidebar.radio(
        "Platform Modules",
        [
            "1. Executive Dashboard",
            "2. Claims Analytics",
            "3. Fraud Risk Scoring",
            "4. Model Monitoring & MLOps",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Architecture")
    st.sidebar.markdown(
        """
        - **Data Lake**: AWS S3
        - **Warehouse**: Snowflake (`INSURANCE_DB`)
        - **Layers**: `RAW` → `STAGING` → `ANALYTICS`
        - **MLOps**: MLflow Model Registry
        - **Serving**: Streamlit + Plotly
        """
    )
    st.sidebar.caption("v1.0.0-production")
    return page


def page_executive_dashboard():
    """Executive KPI Dashboard with Snowflake aggregation metrics."""
    st.markdown('<div class="main-title">Executive Claims Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">High-level enterprise summary of insurance claims exposure, loss metrics, and fraud detection rates.</div>',
        unsafe_allow_html=True,
    )

    # Status indicator
    _, is_live, status_str = load_analytical_base_data()
    if is_live:
        st.success(f"🟢 **Live Connection**: {status_str}")
    else:
        st.info(f"🟠 **Analytical Source**: {status_str} (Set Snowflake credentials to query live warehouse)")

    # KPIs
    kpis = get_kpis()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Claims Incurred</div>
                <div class="metric-card-value">{format_number(kpis['total_claims'])}</div>
                <div class="metric-card-sub">Portfolio size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Claim Payouts</div>
                <div class="metric-card-value">{format_currency(kpis['total_claim_amount'])}</div>
                <div class="metric-card-sub">Incurred losses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Average Claim Severity</div>
                <div class="metric-card-value">{format_currency(kpis['average_claim_amount'])}</div>
                <div class="metric-card-sub">Median: {format_currency(kpis['median_claim_amount'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Detected Fraud Rate</div>
                <div class="metric-card-value" style="color: #dc2626;">{format_percentage(kpis['fraud_rate'])}</div>
                <div class="metric-card-sub">{format_number(kpis['fraudulent_claims'])} flagged claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Avg Annual Premium", format_currency(kpis["average_annual_premium"]))
    with c6:
        st.metric("Avg Insured Age", f"{kpis['average_customer_age']:.1f} yrs")
    with c7:
        est_prevented_fraud = kpis['fraudulent_claims'] * kpis['average_claim_amount'] * 0.70
        st.metric("Potential Fraud Loss Prevented", format_currency(est_prevented_fraud))
    with c8:
        st.metric("Model Operating Threshold", "0.35 (Calibrated)")

    st.markdown("---")

    # Overview Plots
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Claims & Incurred Losses by Incident State")
        df_state = get_claims_by_state()
        fig_state = px.bar(
            df_state,
            x="INCIDENT_STATE",
            y="CLAIM_COUNT",
            color="FRAUD_RATE",
            color_continuous_scale="Blues",
            labels={"CLAIM_COUNT": "Total Claims", "FRAUD_RATE": "Fraud Rate (%)", "INCIDENT_STATE": "State"},
            title="Claim Frequency and Fraud Concentration by Incident State",
        )
        fig_state.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(fig_state, use_container_width=True)

    with col_right:
        st.subheader("Claim Breakdown by Incident Severity")
        df_sev = get_claims_by_severity()
        fig_sev = px.pie(
            df_sev,
            names="INCIDENT_SEVERITY",
            values="CLAIM_COUNT",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Distribution Across Incident Severity Classes",
        )
        fig_sev.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(fig_sev, use_container_width=True)


def page_claims_analytics():
    """Deep dive analytics with interactive Plotly visuals and filters."""
    st.markdown('<div class="main-title">Interactive Claims Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore claims distributions, monthly seasonality, severity correlations, and fraud patterns.</div>',
        unsafe_allow_html=True,
    )

    df_base, _, _ = load_analytical_base_data()

    # Interactive Filters
    st.markdown("##### Filter Claims Cohort")
    fc1, fc2, fc3, fc4 = st.columns(4)

    states = ["All"] + sorted(df_base["INCIDENT_STATE"].dropna().unique().tolist())
    selected_state = fc1.selectbox("Incident State", states)

    incident_types = ["All"] + sorted(df_base["INCIDENT_TYPE"].dropna().unique().tolist())
    selected_type = fc2.selectbox("Incident Type", incident_types)

    severities = ["All"] + sorted(df_base["INCIDENT_SEVERITY"].dropna().unique().tolist())
    selected_sev = fc3.selectbox("Incident Severity", severities)

    fraud_options = ["All", "Fraud Only (1)", "Non-Fraud Only (0)"]
    selected_fraud = fc4.selectbox("Fraud Tag", fraud_options)

    # Filter application
    filtered = df_base.copy()
    if selected_state != "All":
        filtered = filtered[filtered["INCIDENT_STATE"] == selected_state]
    if selected_type != "All":
        filtered = filtered[filtered["INCIDENT_TYPE"] == selected_type]
    if selected_sev != "All":
        filtered = filtered[filtered["INCIDENT_SEVERITY"] == selected_sev]
    if selected_fraud == "Fraud Only (1)":
        filtered = filtered[filtered["FRAUD_FLAG"] == 1]
    elif selected_fraud == "Non-Fraud Only (0)":
        filtered = filtered[filtered["FRAUD_FLAG"] == 0]

    st.caption(f"Showing **{len(filtered):,}** matching claims (out of {len(df_base):,} total).")

    tab1, tab2, tab3 = st.tabs(["Claims & Financials", "Monthly Seasonality", "Fraud Risk Profiles"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                filtered,
                x="POLICY_ANNUAL_PREMIUM",
                y="TOTAL_CLAIM_AMOUNT",
                color="FRAUD_FLAG",
                color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
                labels={"POLICY_ANNUAL_PREMIUM": "Annual Premium ($)", "TOTAL_CLAIM_AMOUNT": "Claim Amount ($)", "FRAUD_FLAG": "Fraud Flag"},
                title="Claim Amount vs. Annual Premium",
                opacity=0.7,
                hover_data=["INCIDENT_TYPE", "INCIDENT_SEVERITY", "CLAIM_TO_PREMIUM_RATIO"],
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            fig_hist = px.histogram(
                filtered,
                x="TOTAL_CLAIM_AMOUNT",
                color="FRAUD_FLAG",
                barmode="overlay",
                nbins=30,
                color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
                labels={"TOTAL_CLAIM_AMOUNT": "Total Claim Amount ($)", "FRAUD_FLAG": "Fraud Flag"},
                title="Claim Amount Distribution by Fraud Status",
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        df_monthly = get_monthly_claims()
        fig_month = px.line(
            df_monthly,
            x="INCIDENT_PERIOD" if "INCIDENT_PERIOD" in df_monthly.columns else df_monthly.columns[0],
            y="TOTAL_CLAIM_AMOUNT",
            markers=True,
            labels={"TOTAL_CLAIM_AMOUNT": "Total Claims Paid ($)", "INCIDENT_PERIOD": "Month"},
            title="Monthly Incurred Losses Trend",
        )
        fig_month.update_traces(line_color="#2563eb", line_width=3)
        fig_month.update_layout(height=400)
        st.plotly_chart(fig_month, use_container_width=True)

    with tab3:
        df_fraud = get_fraud_analysis()
        st.subheader("Statistical Profile: Fraud vs. Non-Fraud Claims")
        st.dataframe(
            df_fraud.style.format({
                "TOTAL_CLAIM_AMOUNT": "${:,.0f}",
                "AVERAGE_CLAIM_AMOUNT": "${:,.2f}",
                "AVERAGE_POLICY_PREMIUM": "${:,.2f}",
                "AVERAGE_CUSTOMER_AGE": "{:.1f}",
                "AVG_VEHICLES_INVOLVED": "{:.2f}",
                "AVERAGE_WITNESSES": "{:.2f}",
                "AVERAGE_BODILY_INJURIES": "{:.2f}",
            }),
            use_container_width=True,
        )

        fig_box = px.box(
            filtered,
            x="INCIDENT_SEVERITY",
            y="TOTAL_CLAIM_AMOUNT",
            color="FRAUD_FLAG",
            color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
            title="Claim Severity vs. Dollar Value by Fraud Tag",
        )
        st.plotly_chart(fig_box, use_container_width=True)


def page_fraud_risk_scoring():
    """Main ML Scoring interface with interactive form, preset loading, and risk categorization."""
    st.markdown('<div class="main-title">Fraud Risk Scoring Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter claim information or load a benchmark case to generate real-time calibrated fraud risk scores.</div>',
        unsafe_allow_html=True,
    )

    sample_cases = get_sample_claims()

    # Preset selector
    c_preset, c_clear = st.columns([3, 1])
    preset_choice = c_preset.selectbox(
        "⚡ Quick-Fill Benchmark Scenarios",
        ["Manual Entry"] + list(sample_cases.keys()),
        index=0,
    )

    preset_data = {}
    if preset_choice in sample_cases:
        preset_data = sample_cases[preset_choice]

    st.markdown("---")

    with st.form("claim_scoring_form"):
        st.subheader("1. Customer & Policy Profile")
        col_c1, col_c2, col_c3 = st.columns(3)
        age = col_c1.number_input("Customer Age", 18, 100, int(preset_data.get("AGE", 38)))
        months = col_c2.number_input("Months as Customer", 0, 600, int(preset_data.get("MONTHS_AS_CUSTOMER", 120)))
        sex = col_c3.selectbox("Insured Sex", ["MALE", "FEMALE"], index=0 if preset_data.get("INSURED_SEX") == "MALE" else 1)

        education_options = ["High School", "College", "Associate", "Bachelor", "Masters", "MD", "PhD"]
        edu = col_c1.selectbox("Education Level", education_options, index=education_options.index(preset_data.get("INSURED_EDUCATION_LEVEL", "Bachelor")) if preset_data.get("INSURED_EDUCATION_LEVEL") in education_options else 3)
        
        occ_options = ["sales", "exec-managerial", "prof-specialty", "tech-support", "craft-repair", "transport-moving", "armed-forces", "machine-op-inspct", "protective-serv", "other-service"]
        occ = col_c2.selectbox("Occupation", occ_options, index=occ_options.index(preset_data.get("INSURED_OCCUPATION", "sales")) if preset_data.get("INSURED_OCCUPATION") in occ_options else 0)
        
        rel_options = ["husband", "wife", "own-child", "unmarried", "not-in-family", "other-relative"]
        rel = col_c3.selectbox("Relationship", rel_options, index=rel_options.index(preset_data.get("INSURED_RELATIONSHIP", "husband")) if preset_data.get("INSURED_RELATIONSHIP") in rel_options else 0)

        col_p1, col_p2, col_p3 = st.columns(3)
        policy_state = col_p1.selectbox("Policy State", ["OH", "IL", "IN"], index=["OH", "IL", "IN"].index(preset_data.get("POLICY_STATE", "OH")) if preset_data.get("POLICY_STATE") in ["OH", "IL", "IN"] else 0)
        csl = col_p2.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"], index=1)
        deductible = col_p3.selectbox("Deductible ($)", [500, 1000, 2000], index=1)

        premium = col_p1.number_input("Annual Premium ($)", 100.0, 5000.0, float(preset_data.get("POLICY_ANNUAL_PREMIUM", 1250.0)), step=50.0)
        umbrella = col_p2.number_input("Umbrella Limit ($)", 0, 10000000, int(preset_data.get("UMBRELLA_LIMIT", 0)), step=1000000)
        policy_tenure = col_p3.number_input("Policy Tenure (Months)", 0, 600, int(preset_data.get("POLICY_TENURE_MONTHS", 120)))

        st.subheader("2. Incident & Loss Details")
        col_i1, col_i2, col_i3 = st.columns(3)
        inc_types = ["Single Vehicle Collision", "Multi-vehicle Collision", "Side Collision", "Parked Car", "Vehicle Theft"]
        inc_type = col_i1.selectbox("Incident Type", inc_types, index=inc_types.index(preset_data.get("INCIDENT_TYPE", "Multi-vehicle Collision")) if preset_data.get("INCIDENT_TYPE") in inc_types else 1)

        col_types = ["Front Collision", "Rear Collision", "Side Collision", "None"]
        collision = col_i2.selectbox("Collision Type", col_types, index=col_types.index(preset_data.get("COLLISION_TYPE", "Rear Collision")) if preset_data.get("COLLISION_TYPE") in col_types else 1)

        severities = ["Minor Damage", "Trivial Damage", "Major Damage", "Total Loss"]
        sev = col_i3.selectbox("Incident Severity", severities, index=severities.index(preset_data.get("INCIDENT_SEVERITY", "Minor Damage")) if preset_data.get("INCIDENT_SEVERITY") in severities else 0)

        authorities = ["Police", "Fire", "Ambulance", "Other", "None"]
        auth = col_i1.selectbox("Authorities Contacted", authorities, index=authorities.index(preset_data.get("AUTHORITIES_CONTACTED", "Police")) if preset_data.get("AUTHORITIES_CONTACTED") in authorities else 0)

        inc_state = col_i2.selectbox("Incident State", ["NY", "SC", "WV", "VA", "NC", "PA", "OH"], index=0)
        inc_hour = col_i3.slider("Hour of Incident (24hr)", 0, 23, int(preset_data.get("INCIDENT_HOUR_OF_THE_DAY", 14)))

        col_d1, col_d2, col_d3 = st.columns(3)
        vehicles = col_d1.number_input("Vehicles Involved", 1, 10, int(preset_data.get("NUMBER_OF_VEHICLES_INVOLVED", 2)))
        injuries = col_d2.number_input("Bodily Injuries", 0, 10, int(preset_data.get("BODILY_INJURIES", 0)))
        witnesses = col_d3.number_input("Witnesses", 0, 10, int(preset_data.get("WITNESSES", 1)))

        police_rep = col_d1.selectbox("Police Report Available", ["YES", "NO"], index=0 if preset_data.get("POLICE_REPORT_AVAILABLE") == "YES" else 1)
        prop_dam = col_d2.selectbox("Property Damage Reported", ["YES", "NO"], index=0 if preset_data.get("PROPERTY_DAMAGE") == "YES" else 1)
        cap_loss = col_d3.number_input("Capital Loss ($)", -100000, 0, int(preset_data.get("CAPITAL_LOSS", 0)), step=5000)

        st.subheader("3. Claim Financials & Vehicle Details")
        col_f1, col_f2, col_f3 = st.columns(3)
        injury_claim = col_f1.number_input("Injury Claim ($)", 0, 100000, int(preset_data.get("INJURY_CLAIM", 5000)), step=1000)
        prop_claim = col_f2.number_input("Property Claim ($)", 0, 100000, int(preset_data.get("PROPERTY_CLAIM", 5000)), step=1000)
        veh_claim = col_f3.number_input("Vehicle Claim ($)", 0, 100000, int(preset_data.get("VEHICLE_CLAIM", 20000)), step=1000)

        tot_claim = injury_claim + prop_claim + veh_claim
        st.info(f"Calculated Total Claim Amount: **${tot_claim:,.2f}** | Claim-to-Premium Ratio: **{(tot_claim / max(premium, 1.0)):.1f}x**")

        auto_make = col_f1.selectbox("Auto Make", ["Saab", "Mercedes", "Dodge", "Chevrolet", "Ford", "BMW", "Toyota", "Volkswagen", "Nissan", "Subaru", "Honda", "Audi", "Accura", "Jeep"], index=0)
        auto_year = col_f2.number_input("Auto Year", 1995, 2026, int(preset_data.get("AUTO_YEAR", 2010)))
        veh_age = 2015 - auto_year

        submitted = st.form_submit_button("🛡️ Score Claim Risk", type="primary", use_container_width=True)

    if submitted:
        claim_payload = {
            "MONTHS_AS_CUSTOMER": months,
            "AGE": age,
            "POLICY_STATE": policy_state,
            "POLICY_CSL": csl,
            "POLICY_DEDUCTIBLE": deductible,
            "POLICY_ANNUAL_PREMIUM": premium,
            "UMBRELLA_LIMIT": umbrella,
            "INSURED_SEX": sex,
            "INSURED_EDUCATION_LEVEL": edu,
            "INSURED_OCCUPATION": occ,
            "INSURED_RELATIONSHIP": rel,
            "POLICY_TENURE_MONTHS": policy_tenure,
            "POLICY_BIND_YEAR": 2014,
            "INCIDENT_TYPE": inc_type,
            "COLLISION_TYPE": None if collision == "None" else collision,
            "INCIDENT_SEVERITY": sev,
            "AUTHORITIES_CONTACTED": auth,
            "INCIDENT_STATE": inc_state,
            "INCIDENT_CITY": "Columbus",
            "INCIDENT_HOUR_OF_THE_DAY": inc_hour,
            "NUMBER_OF_VEHICLES_INVOLVED": vehicles,
            "PROPERTY_DAMAGE": prop_dam,
            "BODILY_INJURIES": injuries,
            "WITNESSES": witnesses,
            "POLICE_REPORT_AVAILABLE": police_rep,
            "CAPITAL_GAINS": 0,
            "CAPITAL_LOSS": cap_loss,
            "TOTAL_CLAIM_AMOUNT": tot_claim,
            "INJURY_CLAIM": injury_claim,
            "PROPERTY_CLAIM": prop_claim,
            "VEHICLE_CLAIM": veh_claim,
            "CLAIM_TO_PREMIUM_RATIO": round(tot_claim / max(premium, 1.0), 2),
            "AUTO_MAKE": auto_make,
            "AUTO_MODEL": "Standard",
            "AUTO_YEAR": auto_year,
            "VEHICLE_AGE": veh_age,
            "INCIDENT_YEAR": 2015,
            "INCIDENT_MONTH": 2,
        }

        with st.spinner("Evaluating through champion ML model pipeline..."):
            result = predict_claim_risk(claim_payload)

        st.markdown("---")
        st.subheader("Fraud Risk Assessment Result")

        res_c1, res_c2 = st.columns([1, 1])

        with res_c1:
            # Risk level badge
            st.markdown(get_risk_badge_html(result["risk_level"]), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Fraud Probability", f"{result['fraud_probability']:.1f}%")

            if result["is_flagged_for_investigation"]:
                st.error(
                    f"🚨 **Action Required**: Score ({result['fraud_probability']:.1f}%) exceeds the business operating threshold ({result['operating_threshold'] * 100:.0f}%). Flagged for Special Investigation Unit (SIU) triage."
                )
            else:
                st.success(
                    f"✅ **Fast-Track Eligible**: Score ({result['fraud_probability']:.1f}%) is beneath investigation threshold. Cleared for standard claim adjudication."
                )

        with res_c2:
            # Plotly Gauge Chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["fraud_probability"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Fraud Probability Index (%)", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0f172a"},
                        "steps": [
                            {"range": [0, result["operating_threshold"] * 100], "color": "#dcfce7"},
                            {"range": [result["operating_threshold"] * 100, 60], "color": "#fef9c3"},
                            {"range": [60, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": result["operating_threshold"] * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        if result["risk_signals"]:
            st.markdown("##### 🔍 Identified Risk Indicators")
            for sig in result["risk_signals"]:
                st.markdown(f"- ⚠️ **{sig}**")


def page_model_monitoring():
    """Model Monitoring and MLOps transparency page."""
    st.markdown('<div class="main-title">Model Monitoring & MLOps</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Live MLflow experiment tracking records, registered champion model status, and threshold trade-offs.</div>',
        unsafe_allow_html=True,
    )

    meta_file = ARTIFACTS_DIR / "champion_metadata.json"
    if not meta_file.exists():
        st.warning("No trained champion model metadata found. Please run `python ml/train.py`.")
        return

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Champion Card
    st.subheader("Champion Model Registry Status")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Registered Champion", meta.get("champion_model_name", "Random Forest"))
    mc2.metric("MLflow Registry Alias", f"@{CHAMPION_ALIAS}")
    mc3.metric("Operating Threshold", f"{meta.get('operating_threshold', DEFAULT_OPERATING_THRESHOLD):.2f}")
    mc4.metric("ROC-AUC Score", f"{meta['metrics']['roc_auc']:.4f}")

    st.markdown("---")

    # Metrics Summary
    st.subheader("Champion Performance on Holdout Test Set (200 Claims)")
    m = meta["metrics"]
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Accuracy", f"{m['accuracy']:.4f}")
    pm2.metric("Fraud Recall", f"{m['fraud_recall']:.4f}", help="Portion of actual fraud caught")
    pm3.metric("Fraud Precision", f"{m['fraud_precision']:.4f}", help="Precision of flagged investigations")
    pm4.metric("Fraud F1 Score", f"{m['fraud_f1']:.4f}")
    pm5.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

    st.markdown("---")

    # Candidate Comparison Table
    st.subheader("MLflow Candidate Model Comparison")
    if "all_model_results" in meta:
        comp_rows = []
        for name, data in meta["all_model_results"].items():
            comp_rows.append({
                "Model Candidate": name,
                "ROC-AUC": round(data["metrics"]["roc_auc"], 4),
                "Fraud Recall": round(data["metrics"]["fraud_recall"], 4),
                "Fraud Precision": round(data["metrics"]["fraud_precision"], 4),
                "Fraud F1": round(data["metrics"]["fraud_f1"], 4),
                "Accuracy": round(data["metrics"]["accuracy"], 4),
                "Threshold": round(data["threshold"], 2),
                "MLflow Run ID": data["run_id"][:8] + "...",
            })
        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_comp.style.highlight_max(subset=["ROC-AUC", "Fraud F1", "Fraud Recall"], color="#dcfce7"), use_container_width=True)

    # Artifact Visualizations
    st.subheader("Evaluation Visual Artifacts (Logged in MLflow)")
    temp_eval_dir = ARTIFACTS_DIR / "temp_eval"

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        roc_img = temp_eval_dir / f"{meta['champion_model_name']}_roc_curve.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Holdout ROC Curve (Discriminative Ability)", use_column_width=True)

        thresh_img = temp_eval_dir / f"{meta['champion_model_name']}_threshold_tradeoff.png"
        if thresh_img.exists():
            st.image(str(thresh_img), caption="Precision vs. Recall vs. Threshold Trade-off", use_column_width=True)

    with vcol2:
        cm_img = temp_eval_dir / f"{meta['champion_model_name']}_confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Holdout Confusion Matrix", use_column_width=True)

        feat_img = temp_eval_dir / f"{meta['champion_model_name']}_feature_importance.png"
        if feat_img.exists():
            st.image(str(feat_img), caption="Top 15 Predictive Features (Gini Importance)", use_column_width=True)


def main():
    """Main application dispatcher."""
    selected_page = render_sidebar()

    if selected_page == "1. Executive Dashboard":
        page_executive_dashboard()
    elif selected_page == "2. Claims Analytics":
        page_claims_analytics()
    elif selected_page == "3. Fraud Risk Scoring":
        page_fraud_risk_scoring()
    elif selected_page == "4. Model Monitoring & MLOps":
        page_model_monitoring()


if __name__ == "__main__":
    main()
