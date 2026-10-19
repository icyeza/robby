-- STAND-IN for a clinic's OpenClinic database. This is NOT the real OpenClinic GA schema (which
-- could not be obtained); it is a plausible maternity-capable layout so the whole flow can be
-- exercised end to end. When a real schema is available, change query.sql, nothing else.

CREATE DATABASE IF NOT EXISTS openclinic CHARACTER SET utf8mb4;
USE openclinic;

CREATE TABLE IF NOT EXISTS patient (
  patient_id    VARCHAR(20) PRIMARY KEY,          -- the hospital file number
  first_name    VARCHAR(80) NOT NULL,
  last_name     VARCHAR(80) NOT NULL,
  date_of_birth DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS encounter (
  encounter_id INT AUTO_INCREMENT PRIMARY KEY,
  patient_id   VARCHAR(20) NOT NULL,
  ward         VARCHAR(40) NOT NULL,
  admission_at DATETIME NOT NULL,                 -- local time
  discharge_at DATETIME NULL,
  FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
  INDEX (ward, admission_at)
);

CREATE TABLE IF NOT EXISTS obstetric_record (
  encounter_id  INT PRIMARY KEY,
  gravida       TINYINT NULL,
  para          TINYINT NULL,                     -- previous births
  previous_cs   TINYINT NULL,
  gest_age_text VARCHAR(30) NULL,                 -- free text: "39", "39+2", "38w4d"
  pregnancy     VARCHAR(20) NULL,                 -- Singleton / Twins / ...
  presentation  VARCHAR(20) NULL,                 -- Vertex / Breech / ...
  labour_onset  VARCHAR(40) NULL,                 -- Spontaneous / Induced / Elective C-section
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id)
);

CREATE TABLE IF NOT EXISTS vitals (
  vitals_id    INT AUTO_INCREMENT PRIMARY KEY,
  encounter_id INT NOT NULL,
  measured_at  DATETIME NOT NULL,
  bp_systolic  SMALLINT NULL,
  bp_diastolic SMALLINT NULL,
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id),
  INDEX (encounter_id, measured_at)
);

CREATE TABLE IF NOT EXISTS lab_result (
  lab_id       INT AUTO_INCREMENT PRIMARY KEY,
  encounter_id INT NOT NULL,
  test_code    VARCHAR(30) NOT NULL,              -- URINE_PROTEIN | GLUCOSE_MGDL
  result_text  VARCHAR(30) NOT NULL,
  resulted_at  DATETIME NOT NULL,
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id),
  INDEX (encounter_id, test_code, resulted_at)
);

-- The connector signs in with a read-only account, as it would at a real clinic.
CREATE USER IF NOT EXISTS 'robby_reader'@'localhost' IDENTIFIED BY 'reader-password-123';
CREATE USER IF NOT EXISTS 'robby_reader'@'127.0.0.1' IDENTIFIED BY 'reader-password-123';
GRANT SELECT ON openclinic.* TO 'robby_reader'@'localhost';
GRANT SELECT ON openclinic.* TO 'robby_reader'@'127.0.0.1';
