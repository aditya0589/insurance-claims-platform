-- ============================================================
-- INSURANCE CLAIMS INTELLIGENCE PLATFORM
-- Snowflake Initial Setup
-- Phase 1: AWS S3 → Snowflake RAW Data ingestion
-- ============================================================


-- ============================================================
-- 1. DATABASE AND SCHEMAS
-- ============================================================

CREATE DATABASE IF NOT EXISTS INSURANCE_DB;

USE DATABASE INSURANCE_DB;

CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS STAGING;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;


-- ============================================================
-- 2. STORAGE INTEGRATION
-- Connect Snowflake to AWS S3 through an IAM Role
-- ============================================================

CREATE OR REPLACE STORAGE INTEGRATION insurance_s3_integration
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = S3
    ENABLED = TRUE
    STORAGE_AWS_ROLE_ARN =
        'arn:aws:iam::<AWS_ACCOUNT_ID_REMOVED>:role/aditya_snowflake_role'
    STORAGE_ALLOWED_LOCATIONS = (
        's3://aditya-project-bucket/snowflake_project/raw/'
    );


-- ============================================================
-- 3. FILE FORMAT
-- Insurance claims CSV
-- ============================================================

USE SCHEMA RAW;

CREATE OR REPLACE FILE FORMAT insurance_csv_format
    TYPE = CSV
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('NULL', 'null', '')
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE;


-- ============================================================
-- 4. EXTERNAL STAGE
-- Points to the raw data stored in S3
-- ============================================================

CREATE OR REPLACE STAGE insurance_s3_stage
    URL = 's3://aditya-project-bucket/snowflake_project/raw/'
    STORAGE_INTEGRATION = insurance_s3_integration;


-- ============================================================
-- 5. VERIFY S3 CONNECTION
-- ============================================================

LIST @insurance_s3_stage;


-- ============================================================
-- 6. RAW INSURANCE CLAIMS TABLE
-- ============================================================

CREATE OR REPLACE TABLE RAW_INSURANCE_CLAIMS (
    months_as_customer INTEGER,
    age INTEGER,
    policy_number INTEGER,
    policy_bind_date DATE,
    policy_state VARCHAR,
    policy_csl VARCHAR,
    policy_deductable INTEGER,
    policy_annual_premium DECIMAL(12,2),
    umbrella_limit INTEGER,
    insured_zip INTEGER,
    insured_sex VARCHAR,
    insured_education_level VARCHAR,
    insured_occupation VARCHAR,
    insured_hobbies VARCHAR,
    insured_relationship VARCHAR,
    capital_gains INTEGER,
    capital_loss INTEGER,
    incident_date DATE,
    incident_type VARCHAR,
    collision_type VARCHAR,
    incident_severity VARCHAR,
    authorities_contacted VARCHAR,
    incident_state VARCHAR,
    incident_city VARCHAR,
    incident_location VARCHAR,
    incident_hour_of_the_day INTEGER,
    number_of_vehicles_involved INTEGER,
    property_damage VARCHAR,
    bodily_injuries INTEGER,
    witnesses INTEGER,
    police_report_available VARCHAR,
    total_claim_amount INTEGER,
    injury_claim INTEGER,
    property_claim INTEGER,
    vehicle_claim INTEGER,
    auto_make VARCHAR,
    auto_model VARCHAR,
    auto_year INTEGER,
    fraud_reported VARCHAR
);


-- ============================================================
-- 7. LOAD DATA FROM S3 → RAW
-- ============================================================

COPY INTO RAW_INSURANCE_CLAIMS
FROM @insurance_s3_stage/insurance_claims.csv
FILE_FORMAT = insurance_csv_format
ON_ERROR = 'ABORT_STATEMENT';


-- ============================================================
-- 8. VERIFY DATA LOAD
-- ============================================================

SELECT COUNT(*) AS TOTAL_RECORDS
FROM RAW_INSURANCE_CLAIMS;


SELECT *
FROM RAW_INSURANCE_CLAIMS
LIMIT 10;
