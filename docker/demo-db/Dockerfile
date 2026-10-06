# The demo database: PostgreSQL with the synthetic e-commerce data generated on first start.
FROM postgres:16

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-numpy \
    && rm -rf /var/lib/apt/lists/*

COPY generate.py schema.sql roles.sql /demo-db/
COPY init-demo.sh /docker-entrypoint-initdb.d/10-init-demo.sh
