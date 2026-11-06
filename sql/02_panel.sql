CREATE OR REPLACE TABLE unit_month AS
SELECT make, model, model_year, system_norm AS system,
       date_trunc('month', received_date)::DATE AS month,
       COUNT(DISTINCT ODINO) AS n,
       COUNT(DISTINCT CASE WHEN crash THEN ODINO END) AS n_crash,
       COUNT(DISTINCT CASE WHEN fire THEN ODINO END) AS n_fire,
       COUNT(DISTINCT CASE WHEN injured > 0 OR deaths > 0 THEN ODINO END) AS n_injury,
       COUNT(DISTINCT CASE WHEN towed THEN ODINO END) AS n_towed
FROM complaints
WHERE system_norm NOT IN ('UNKNOWN OR OTHER', 'OTHER', '0', '')
GROUP BY ALL;

CREATE OR REPLACE TABLE unit_first AS
SELECT make, model, model_year, system, MIN(month) AS first_month
FROM unit_month GROUP BY ALL;

CREATE OR REPLACE TABLE first_recall AS
SELECT make, model, model_year, system_norm AS system,
       MIN(recall_date) AS first_recall,
       arg_min(initiated_by, recall_date) AS initiated_by,
       arg_min(campaign, recall_date) AS campaign
FROM recalls GROUP BY ALL;

CREATE OR REPLACE TABLE family_recall AS
SELECT make, model, system_norm AS system, MIN(recall_date) AS family_first_recall
FROM recalls GROUP BY ALL;

CREATE OR REPLACE TABLE mmy_month AS
SELECT make, model, model_year, date_trunc('month', received_date)::DATE AS month,
       COUNT(DISTINCT ODINO) AS n
FROM complaints GROUP BY ALL;

CREATE OR REPLACE TABLE mms_month AS
SELECT make, model, system_norm AS system, date_trunc('month', received_date)::DATE AS month,
       COUNT(DISTINCT ODINO) AS n
FROM complaints
WHERE system_norm NOT IN ('UNKNOWN OR OTHER', 'OTHER', '0', '')
GROUP BY ALL;

CREATE OR REPLACE TABLE spine AS
SELECT DISTINCT make, model, model_year, system,
       (month + to_months(k::INTEGER))::DATE AS month
FROM unit_month CROSS JOIN range(0, 12) t(k);

CREATE OR REPLACE TABLE spine AS
SELECT s.*
FROM spine s
LEFT JOIN first_recall r
  ON r.make = s.make AND r.model = s.model AND r.model_year = s.model_year AND r.system = s.system
WHERE s.month >= DATE '2016-01-01'
  AND s.month + INTERVAL 7 MONTH <= (SELECT MAX(received_date) FROM complaints)
  AND (r.first_recall IS NULL OR r.first_recall >= s.month + INTERVAL 1 MONTH);

CREATE OR REPLACE TABLE unit_feats AS
SELECT s.make, s.model, s.model_year, s.system, s.month,
       SUM(CASE WHEN u.month = s.month THEN u.n ELSE 0 END) AS n_1m,
       SUM(CASE WHEN u.month > s.month - INTERVAL 3 MONTH THEN u.n ELSE 0 END) AS n_3m,
       SUM(CASE WHEN u.month > s.month - INTERVAL 6 MONTH THEN u.n ELSE 0 END) AS n_6m,
       SUM(u.n) AS n_12m,
       SUM(CASE WHEN u.month <= s.month - INTERVAL 3 MONTH
                 AND u.month > s.month - INTERVAL 6 MONTH THEN u.n ELSE 0 END) AS n_prev3m,
       SUM(u.n_crash) AS crash_12m,
       SUM(u.n_fire) AS fire_12m,
       SUM(u.n_injury) AS injury_12m,
       SUM(u.n_towed) AS towed_12m
FROM spine s
JOIN unit_month u
  ON u.make = s.make AND u.model = s.model AND u.model_year = s.model_year AND u.system = s.system
 AND u.month <= s.month AND u.month > s.month - INTERVAL 12 MONTH
GROUP BY ALL;

CREATE OR REPLACE TABLE ctx_mmy AS
SELECT s.make, s.model, s.model_year, s.month, SUM(m.n) AS vehicle_12m
FROM (SELECT DISTINCT make, model, model_year, month FROM spine) s
JOIN mmy_month m
  ON m.make = s.make AND m.model = s.model AND m.model_year = s.model_year
 AND m.month <= s.month AND m.month > s.month - INTERVAL 12 MONTH
GROUP BY ALL;

CREATE OR REPLACE TABLE ctx_mms AS
SELECT s.make, s.model, s.system, s.month, SUM(m.n) AS model_system_12m
FROM (SELECT DISTINCT make, model, system, month FROM spine) s
JOIN mms_month m
  ON m.make = s.make AND m.model = s.model AND m.system = s.system
 AND m.month <= s.month AND m.month > s.month - INTERVAL 12 MONTH
GROUP BY ALL;

CREATE OR REPLACE TABLE panel AS
SELECT f.*,
       YEAR(f.month) - f.model_year                         AS vehicle_age,
       date_diff('month', uf.first_month, f.month)          AS months_since_first,
       c1.vehicle_12m,
       f.n_12m * 1.0 / NULLIF(c1.vehicle_12m, 0)            AS system_share,
       c2.model_system_12m - f.n_12m                        AS sibling_years_12m,
       (fr.family_first_recall IS NOT NULL
        AND fr.family_first_recall < f.month + INTERVAL 1 MONTH)::INT AS family_recalled,
       r.first_recall, r.initiated_by, r.campaign,
       (r.first_recall IS NOT NULL
        AND r.first_recall < f.month + INTERVAL 7 MONTH)::INT AS target
FROM unit_feats f
JOIN unit_first uf USING (make, model, model_year, system)
LEFT JOIN ctx_mmy c1 USING (make, model, model_year, month)
LEFT JOIN ctx_mms c2 USING (make, model, system, month)
LEFT JOIN family_recall fr USING (make, model, system)
LEFT JOIN first_recall r USING (make, model, model_year, system)