-- ============================================================
-- INSURANCE CLAIMS INTELLIGENCE PLATFORM
-- Phase 4: ANALYTICS
-- ============================================================

USE DATABASE INSURANCE_DB;
USE SCHEMA ANALYTICS;

-- 1. Fact table: one row = one insurance claim / incident
CREATE OR REPLACE TABLE FACT_CLAIMS AS
SELECT *
FROM INSURANCE_DB.STAGING.STG_INSURANCE_CLAIMS;

-- 2. Analytics view with derived features
CREATE OR REPLACE VIEW VW_CLAIMS_ANALYTICS AS
SELECT
    *,
    DATEDIFF('month', policy_bind_date, incident_date) AS policy_tenure_months,
    YEAR(policy_bind_date) AS policy_bind_year,
    YEAR(incident_date) AS incident_year,
    MONTH(incident_date) AS incident_month,
    YEAR(incident_date) - auto_year AS vehicle_age,
    injury_claim + property_claim + vehicle_claim AS calculated_claim_total,
    CASE
        WHEN incident_severity = 'Major Damage' THEN 3
        WHEN incident_severity = 'Total Loss' THEN 4
        WHEN incident_severity = 'Minor Damage' THEN 1
        ELSE 2
    END AS severity_score,
    CASE
        WHEN fraud_reported = 'Y' THEN 1
        WHEN fraud_reported = 'N' THEN 0
        ELSE NULL
    END AS fraud_flag
FROM FACT_CLAIMS;

-- 3. Overall business KPIs
CREATE OR REPLACE VIEW VW_CLAIM_KPIS AS
SELECT
    COUNT(*) AS total_claims,
    SUM(total_claim_amount) AS total_claim_amount,
    AVG(total_claim_amount) AS average_claim_amount,
    MEDIAN(total_claim_amount) AS median_claim_amount,
    SUM(IFF(fraud_reported = 'Y', 1, 0)) AS fraudulent_claims,
    ROUND(100.0 * SUM(IFF(fraud_reported = 'Y', 1, 0)) / COUNT(*), 2) AS fraud_rate,
    AVG(policy_annual_premium) AS average_annual_premium,
    AVG(age) AS average_customer_age
FROM FACT_CLAIMS;

-- 4. Claims by state
CREATE OR REPLACE VIEW VW_CLAIMS_BY_STATE AS
SELECT
    incident_state,
    COUNT(*) AS claim_count,
    SUM(total_claim_amount) AS total_claim_amount,
    AVG(total_claim_amount) AS average_claim_amount,
    SUM(IFF(fraud_reported = 'Y', 1, 0)) AS fraudulent_claims,
    ROUND(100.0 * SUM(IFF(fraud_reported = 'Y', 1, 0)) / COUNT(*), 2) AS fraud_rate
FROM FACT_CLAIMS
GROUP BY incident_state
ORDER BY claim_count DESC;

-- 5. Claims by incident type
CREATE OR REPLACE VIEW VW_CLAIMS_BY_INCIDENT_TYPE AS
SELECT
    incident_type,
    COUNT(*) AS claim_count,
    SUM(total_claim_amount) AS total_claim_amount,
    AVG(total_claim_amount) AS average_claim_amount,
    SUM(IFF(fraud_reported = 'Y', 1, 0)) AS fraudulent_claims
FROM FACT_CLAIMS
GROUP BY incident_type
ORDER BY claim_count DESC;

-- 6. Claims by incident severity
CREATE OR REPLACE VIEW VW_CLAIMS_BY_SEVERITY AS
SELECT
    incident_severity,
    COUNT(*) AS claim_count,
    SUM(total_claim_amount) AS total_claim_amount,
    AVG(total_claim_amount) AS average_claim_amount,
    SUM(IFF(fraud_reported = 'Y', 1, 0)) AS fraudulent_claims,
    ROUND(100.0 * SUM(IFF(fraud_reported = 'Y', 1, 0)) / COUNT(*), 2) AS fraud_rate
FROM FACT_CLAIMS
GROUP BY incident_severity
ORDER BY claim_count DESC;

-- 7. Monthly claim trends
CREATE OR REPLACE VIEW VW_MONTHLY_CLAIMS AS
SELECT
    DATE_TRUNC('month', incident_date) AS incident_month,
    COUNT(*) AS claim_count,
    SUM(total_claim_amount) AS total_claim_amount,
    AVG(total_claim_amount) AS average_claim_amount,
    SUM(IFF(fraud_reported = 'Y', 1, 0)) AS fraudulent_claims
FROM FACT_CLAIMS
GROUP BY DATE_TRUNC('month', incident_date)
ORDER BY incident_month;

-- 8. Fraud analysis
CREATE OR REPLACE VIEW VW_FRAUD_ANALYSIS AS
SELECT
    fraud_reported,
    COUNT(*) AS claim_count,
    SUM(total_claim_amount) AS total_claim_amount,
    AVG(total_claim_amount) AS average_claim_amount,
    AVG(policy_annual_premium) AS average_policy_premium,
    AVG(age) AS average_customer_age,
    AVG(number_of_vehicles_involved) AS avg_vehicles_involved,
    AVG(witnesses) AS average_witnesses,
    AVG(bodily_injuries) AS average_bodily_injuries
FROM FACT_CLAIMS
GROUP BY fraud_reported
ORDER BY fraud_reported;

-- 9. Validation / output queries
SELECT * FROM VW_CLAIM_KPIS;
SELECT * FROM VW_FRAUD_ANALYSIS;
SELECT * FROM VW_CLAIMS_BY_STATE;
SELECT * FROM VW_CLAIMS_BY_INCIDENT_TYPE;
SELECT * FROM VW_CLAIMS_BY_SEVERITY;
SELECT * FROM VW_MONTHLY_CLAIMS;
