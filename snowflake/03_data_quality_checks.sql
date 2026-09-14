-- ============================================================
-- INSURANCE CLAIMS INTELLIGENCE PLATFORM
-- Data Quality Checks
--
-- Purpose:
--   Validate the STAGING layer before the data is used
--   for analytics.
-- ============================================================


USE DATABASE INSURANCE_DB;
USE SCHEMA STAGING;


-- ============================================================
-- 1. RECORD COUNT
-- ============================================================

SELECT
    COUNT(*) AS RECORD_COUNT
FROM STG_INSURANCE_CLAIMS;


-- ============================================================
-- 2. INVALID AGES
-- ============================================================

SELECT
    COUNT(*) AS INVALID_AGES
FROM STG_INSURANCE_CLAIMS
WHERE age < 18
   OR age > 100;


-- ============================================================
-- 3. INVALID POLICY PREMIUMS
-- ============================================================

SELECT
    COUNT(*) AS INVALID_PREMIUMS
FROM STG_INSURANCE_CLAIMS
WHERE policy_annual_premium <= 0;


-- ============================================================
-- 4. INVALID CLAIM AMOUNTS
-- ============================================================

SELECT
    COUNT(*) AS INVALID_CLAIMS
FROM STG_INSURANCE_CLAIMS
WHERE total_claim_amount <= 0;


-- ============================================================
-- 5. MISSING COLLISION TYPE
-- ============================================================

SELECT
    COUNT(*) AS MISSING_COLLISION_TYPE
FROM STG_INSURANCE_CLAIMS
WHERE collision_type IS NULL;


-- ============================================================
-- 6. MISSING POLICE REPORT INFORMATION
-- ============================================================

SELECT
    COUNT(*) AS MISSING_POLICE_REPORT
FROM STG_INSURANCE_CLAIMS
WHERE police_report_available IS NULL;


-- ============================================================
-- 7. COLLISION TYPE DISTRIBUTION
-- ============================================================

SELECT
    collision_type,
    COUNT(*) AS claim_count
FROM STG_INSURANCE_CLAIMS
GROUP BY collision_type
ORDER BY claim_count DESC;


-- ============================================================
-- 8. POLICE REPORT DISTRIBUTION
-- ============================================================

SELECT
    police_report_available,
    COUNT(*) AS claim_count
FROM STG_INSURANCE_CLAIMS
GROUP BY police_report_available
ORDER BY claim_count DESC;


-- ============================================================
-- 9. FRAUD DISTRIBUTION
-- ============================================================

SELECT
    fraud_reported,
    COUNT(*) AS claim_count
FROM STG_INSURANCE_CLAIMS
GROUP BY fraud_reported
ORDER BY fraud_reported;


-- ============================================================
-- 10. DATA QUALITY SUMMARY VIEW
-- ============================================================

CREATE OR REPLACE VIEW VW_DATA_QUALITY AS

SELECT
    'INVALID_AGE' AS CHECK_NAME,
    COUNT(*) AS FAILED_RECORDS
FROM STG_INSURANCE_CLAIMS
WHERE age < 18
   OR age > 100

UNION ALL

SELECT
    'INVALID_PREMIUM',
    COUNT(*)
FROM STG_INSURANCE_CLAIMS
WHERE policy_annual_premium <= 0

UNION ALL

SELECT
    'INVALID_CLAIM_AMOUNT',
    COUNT(*)
FROM STG_INSURANCE_CLAIMS
WHERE total_claim_amount <= 0

UNION ALL

SELECT
    'MISSING_COLLISION_TYPE',
    COUNT(*)
FROM STG_INSURANCE_CLAIMS
WHERE collision_type IS NULL

UNION ALL

SELECT
    'MISSING_POLICE_REPORT',
    COUNT(*)
FROM STG_INSURANCE_CLAIMS
WHERE police_report_available IS NULL;


-- ============================================================
-- 11. VIEW DATA QUALITY SUMMARY
-- ============================================================

SELECT *
FROM VW_DATA_QUALITY;
