#!/bin/sh
# Runs once, on the first start of the container (docker-entrypoint-initdb.d).
# Builds the schema, loads the generated data, creates the two roles.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f /demo-db/schema.sql
python3 /demo-db/generate.py --customers "${DEMO_CUSTOMERS:-10000}" --seed "${DEMO_SEED:-7}" --copy |
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v ro_password="${DEMO_RO_PASSWORD:-mlpilot_demo_ro}" \
    -v rw_password="${DEMO_RW_PASSWORD:-mlpilot_demo_rw}" \
    -f /demo-db/roles.sql
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "ANALYZE"
