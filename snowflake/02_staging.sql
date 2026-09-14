-- ============================================================
-- INSURANCE CLAIMS INTELLIGENCE PLATFORM
-- Phase 2: RAW → STAGING
--
-- Purpose:
--   Transform the raw insurance claims data into a cleaned
--   and standardized staging table.
--
-- Source:
--   INSURANCE_DB.RAW.RAW_INSURANCE_CLAIMS
--
-- Target:
--   INSURANCE_DB.STAGING.STG_INSURANCE_CLAIMS
--
-- Key transformations:
--   1. Convert '?' missing values to NULL
--   2. Standardize categorical values
--   3. Rename policy_deductable → policy_deductible
--   4. Preserve the original RAW table unchanged
-- ============================================================


-- ============================================================
-- 1. DATABASE AND SCHEMA
-- ============================================================

USE DATABASE INSURANCE_DB;
USE SCHEMA STAGING;


-- ============================================================
-- 2. CREATE STAGING TABLE
-- ============================================================

CREATE OR REPLACE TABLE STG_INSURANCE_CLAIMS AS

SELECT

    -- --------------------------------------------------------
    -- Policy Information
    -- --------------------------------------------------------

    months_as_customer,

    age,

    policy_number,

    policy_bind_date,

    NULLIF(TRIM(policy_state), '?') AS policy_state,

    NULLIF(TRIM(policy_csl), '?') AS policy_csl,

    -- Correct spelling of source column
    policy_deductable AS policy_deductible,

    policy_annual_premium,

    umbrella_limit,

    insured_zip,


    -- --------------------------------------------------------
    -- Customer Information
    -- --------------------------------------------------------

    NULLIF(
        UPPER(TRIM(insured_sex)),
        '?'
    ) AS insured_sex,

    NULLIF(
        TRIM(insured_education_level),
        '?'
    ) AS insured_education_level,

    NULLIF(
        TRIM(insured_occupation),
        '?'
    ) AS insured_occupation,

    NULLIF(
        TRIM(insured_hobbies),
        '?'
    ) AS insured_hobbies,

    NULLIF(
        TRIM(insured_relationship),
        '?'
    ) AS insured_relationship,


    -- --------------------------------------------------------
    -- Financial Information
    -- --------------------------------------------------------

    capital_gains,

    capital_loss,


    -- --------------------------------------------------------
    -- Incident Information
    -- --------------------------------------------------------

    incident_date,

    NULLIF(
        TRIM(incident_type),
        '?'
    ) AS incident_type,

    NULLIF(
        TRIM(collision_type),
        '?'
    ) AS collision_type,

    NULLIF(
        TRIM(incident_severity),
        '?'
    ) AS incident_severity,

    NULLIF(
        TRIM(authorities_contacted),
        '?'
    ) AS authorities_contacted,

    NULLIF(
        TRIM(incident_state),
        '?'
    ) AS incident_state,

    NULLIF(
        TRIM(incident_city),
        '?'
    ) AS incident_city,

    NULLIF(
        TRIM(incident_location),
        '?'
    ) AS incident_location,

    incident_hour_of_the_day,

    number_of_vehicles_involved,


    -- --------------------------------------------------------
    -- Damage and Investigation Information
    -- --------------------------------------------------------

    NULLIF(
        UPPER(TRIM(property_damage)),
        '?'
    ) AS property_damage,

    bodily_injuries,

    witnesses,

    NULLIF(
        UPPER(TRIM(police_report_available)),
        '?'
    ) AS police_report_available,


    -- --------------------------------------------------------
    -- Claim Information
    -- --------------------------------------------------------

    total_claim_amount,

    injury_claim,

    property_claim,

    vehicle_claim,


    -- --------------------------------------------------------
    -- Vehicle Information
    -- --------------------------------------------------------

    NULLIF(
        TRIM(auto_make),
        '?'
    ) AS auto_make,

    NULLIF(
        TRIM(auto_model),
        '?'
    ) AS auto_model,

    auto_year,


    -- --------------------------------------------------------
    -- Fraud Target
    -- --------------------------------------------------------

    NULLIF(
        UPPER(TRIM(fraud_reported)),
        '?'
    ) AS fraud_reported


FROM INSURANCE_DB.RAW.RAW_INSURANCE_CLAIMS;


SELECT COUNT(*) AS RECORD_COUNT
FROM STAGING.STG_INSURANCE_CLAIMS;

SELECT *
FROM STAGING.STG_INSURANCE_CLAIMS
LIMIT 10;
