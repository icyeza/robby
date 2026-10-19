-- The only thing that knows OpenClinic's table layout. The connector runs this and sends each row
-- to the app. The column ALIASES (the quoted names) are the OpenClinic export headers the app
-- understands (see src/robby/clinical/openclinic.py), so they must stay as they are; only the
-- FROM/JOIN/WHERE part changes for another schema. :since is replaced by the lookback date.
--
-- STAND-IN SCHEMA: written for scripts/openclinic_standin/schema.sql, not for real OpenClinic GA.
-- Blood pressure and the lab values are taken at admission (the first reading).
SELECT
  p.patient_id                                   AS `Patient ID`,
  p.first_name                                   AS `First name`,
  p.last_name                                    AS `Last name`,
  DATE_FORMAT(p.date_of_birth, '%d/%m/%Y')       AS `Date of birth`,
  DATE_FORMAT(e.admission_at, '%d/%m/%Y %H:%i')  AS `Admission date`,
  o.para                                         AS `Para`,
  o.previous_cs                                  AS `Previous C-sections`,
  o.gest_age_text                                AS `Gestational age`,
  o.pregnancy                                    AS `Pregnancy type`,
  o.presentation                                 AS `Presentation`,
  o.labour_onset                                 AS `Labour onset`,
  CASE WHEN v.bp_systolic IS NOT NULL
       THEN CONCAT(v.bp_systolic, '/', v.bp_diastolic) END AS `Blood pressure`,
  lp.result_text                                 AS `Urine protein`,
  lg.result_text                                 AS `Glucose (mg/dL)`
FROM encounter e
JOIN patient p            ON p.patient_id = e.patient_id
JOIN obstetric_record o   ON o.encounter_id = e.encounter_id
LEFT JOIN vitals v        ON v.vitals_id = (
    SELECT vitals_id FROM vitals WHERE encounter_id = e.encounter_id
    ORDER BY measured_at, vitals_id LIMIT 1)
LEFT JOIN lab_result lp   ON lp.lab_id = (
    SELECT lab_id FROM lab_result WHERE encounter_id = e.encounter_id AND test_code = 'URINE_PROTEIN'
    ORDER BY resulted_at, lab_id LIMIT 1)
LEFT JOIN lab_result lg   ON lg.lab_id = (
    SELECT lab_id FROM lab_result WHERE encounter_id = e.encounter_id AND test_code = 'GLUCOSE_MGDL'
    ORDER BY resulted_at, lab_id LIMIT 1)
WHERE e.ward = 'Maternity'
  AND e.admission_at >= :since
ORDER BY e.admission_at, e.encounter_id
