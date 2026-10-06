-- Two roles for the demo database (issue #47). Run by init-demo.sh with psql variables:
--   psql -v ro_password=... -v rw_password=... -f roles.sql
-- mlpilot_ro can only read; mlpilot_rw can also write, to test the "this role can write"
-- warning of the connection test (#43, #44).

CREATE ROLE mlpilot_ro LOGIN PASSWORD :'ro_password';
CREATE ROLE mlpilot_rw LOGIN PASSWORD :'rw_password';

GRANT CONNECT ON DATABASE demo TO mlpilot_ro, mlpilot_rw;
GRANT USAGE ON SCHEMA public TO mlpilot_ro, mlpilot_rw;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mlpilot_ro, mlpilot_rw;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mlpilot_rw;
