-- ============================================================
-- INSURANCE CLAIMS INTELLIGENCE PLATFORM
-- Phase 5: FRAUD DETECTION FEATURE ENGINEERING
--
-- Purpose:
--   Create a clean, ML-ready feature dataset from the analytics
--   layer. Model-specific encoding/scaling will be handled later
--   in Python.
--
-- Source:
--   INSURANCE_DB.ANALYTICS.VW_CLAIMS_ANALYTICS
--
-- Target:
--   INSURANCE_DB.ANALYTICS.FRAUD_MODEL_DATASET
--
-- Grain:
--   One row = one insurance claim / incident
-- ============================================================

USE DATABASE INSURANCE_DB;
USE SCHEMA ANALYTICS;


-- ============================================================
-- 1. CREATE ML-READY FEATURE DATASET
-- ============================================================

CREATE OR REPLACE TABLE FRAUD_MODEL_DATASET AS
SELECT

    -- Policy / customer features
    months_as_customer,
    age,
    policy_state,
    policy_csl,
    policy_deductible,
    policy_annual_premium,
    umbrella_limit,
    insured_sex,
    insured_education_level,
    insured_occupation,
    insured_relationship,

    -- Derived policy features
    policy_tenure_months,
    policy_bind_year,

    -- Incident features
    incident_type,
    collision_type,
    incident_severity,
    authorities_contacted,
    incident_state,
    incident_city,
    incident_hour_of_the_day,
    number_of_vehicles_involved,
    property_damage,
    bodily_injuries,
    witnesses,
    police_report_available,

    -- Financial features
    capital_gains,
    capital_loss,
    total_claim_amount,
    injury_claim,
    property_claim,
    vehicle_claim,

    -- Derived financial feature
    ROUND(
        total_claim_amount / NULLIF(policy_annual_premium, 0),
        2
    ) AS claim_to_premium_ratio,

    -- Vehicle features
    auto_make,
    auto_model,
    auto_year,
    vehicle_age,

    -- Derived analytical features
    incident_year,
    incident_month,

    -- Target variable
    fraud_flag

FROM VW_CLAIMS_ANALYTICS;


-- ============================================================
-- 2. VALIDATE DATASET SIZE
-- ============================================================

SELECT
    COUNT(*) AS total_records
FROM FRAUD_MODEL_DATASET;


-- ============================================================
-- 3. CHECK TARGET DISTRIBUTION
-- ============================================================

SELECT
    fraud_flag,
    COUNT(*) AS claim_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_claims
FROM FRAUD_MODEL_DATASET
GROUP BY fraud_flag
ORDER BY fraud_flag;


-- ============================================================
-- 4. CHECK NULL VALUES IN IMPORTANT FEATURES
-- ============================================================

SELECT
    COUNT_IF(collision_type IS NULL) AS missing_collision_type,
    COUNT_IF(property_damage IS NULL) AS missing_property_damage,
    COUNT_IF(police_report_available IS NULL) AS missing_police_report,
    COUNT_IF(incident_severity IS NULL) AS missing_incident_severity,
    COUNT_IF(claim_to_premium_ratio IS NULL) AS missing_claim_to_premium_ratio
FROM FRAUD_MODEL_DATASET;


-- ============================================================
-- 5. FEATURE SUMMARY
-- ============================================================

SELECT
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    AVG(age) AS avg_age,

    MIN(total_claim_amount) AS min_claim_amount,
    MAX(total_claim_amount) AS max_claim_amount,
    AVG(total_claim_amount) AS avg_claim_amount,

    MIN(claim_to_premium_ratio) AS min_claim_to_premium_ratio,
    MAX(claim_to_premium_ratio) AS max_claim_to_premium_ratio,
    AVG(claim_to_premium_ratio) AS avg_claim_to_premium_ratio,

    MIN(vehicle_age) AS min_vehicle_age,
    MAX(vehicle_age) AS max_vehicle_age,
    AVG(vehicle_age) AS avg_vehicle_age

FROM FRAUD_MODEL_DATASET;


-- ============================================================
-- 6. FRAUD VS NON-FRAUD FEATURE COMPARISON
-- ============================================================

SELECT
    fraud_flag,

    COUNT(*) AS claim_count,

    ROUND(AVG(total_claim_amount), 2)
        AS avg_claim_amount,

    ROUND(AVG(claim_to_premium_ratio), 2)
        AS avg_claim_to_premium_ratio,

    ROUND(AVG(number_of_vehicles_involved), 2)
        AS avg_vehicles_involved,

    ROUND(AVG(witnesses), 2)
        AS avg_witnesses,

    ROUND(AVG(bodily_injuries), 2)
        AS avg_bodily_injuries,

    ROUND(AVG(vehicle_age), 2)
        AS avg_vehicle_age,

    ROUND(AVG(policy_tenure_months), 2)
        AS avg_policy_tenure_months

FROM FRAUD_MODEL_DATASET
GROUP BY fraud_flag
ORDER BY fraud_flag;


-- ============================================================
-- 7. FINAL SAMPLE
-- ============================================================

SELECT *
FROM FRAUD_MODEL_DATASET
LIMIT 10;
