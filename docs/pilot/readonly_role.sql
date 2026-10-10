-- MLPilot pilot kit: a read-only login role for PostgreSQL.
--
-- Run it as a role that may create roles and grant on the tables (your DBA, not MLPilot), with
-- psql 10 or newer, connected to the database MLPilot should read:
--
--   psql "postgresql://dba@db.example.com:5432/shop" \
--        -v pilot_password='choose-a-long-random-password' \
--        -v pilot_schemas='public' \
--        -v pilot_tables='public.customers, public.orders, public.order_items, public.products' \
--        -f readonly_role.sql
--
-- What it creates: one login role that can connect to this database, see the listed schemas, and
-- SELECT from the listed tables and nothing else. It cannot create objects, cannot write, and
-- is not a superuser. A new session of this role starts read-only, with a statement timeout, an
-- idle-in-transaction timeout and a connection limit.
--
-- Variables (set with -v; a variable you leave out takes the default shown):
--   pilot_password           required; the new role's password
--   pilot_tables             required; the tables MLPilot may read, schema-qualified, separated
--                            by commas. Only these are granted. Not "all tables".
--   pilot_schemas            default 'public'; the schemas that hold those tables (USAGE only)
--   pilot_role               default 'mlpilot_pilot'
--   pilot_statement_timeout  default '60s'
--   pilot_idle_timeout       default '60s'  (idle_in_transaction_session_timeout)
--   pilot_connection_limit   default 4
--   revoke                   default 'no'. 'lock' stops the role logging in and ends its sessions;
--                            'drop' also removes its grants and the role. See the end of the file.
--
-- Notes
--   * pilot_schemas and pilot_tables are pasted into the statements as written, so they are
--     trusted input from the person running the script.
--   * The script can be run again (to rotate the password, or after changing pilot_tables):
--     it removes the role's table grants in pilot_schemas first, then grants the list again.
--   * The password ends up in the server log if log_statement is 'ddl' or 'all'. To avoid that,
--     leave pilot_password out and set it afterwards with psql's \password command; the role is
--     then created without one and cannot log in until you do.
--   * The settings below are defaults for new sessions. Whoever has the password can still run
--     SET statement_timeout = 0 or BEGIN READ WRITE. What stops a write is that the role has no
--     write privilege on any table; the defaults are a second layer, not the main one.
--   * PostgreSQL 14 and older give every role CREATE on the schema "public". If you grant on
--     "public" there, also run: REVOKE CREATE ON SCHEMA public FROM PUBLIC;   (this affects all
--     roles, so the script does not do it for you). PostgreSQL 15 and newer do not need it.
--   * Any role can read the system catalogs, so it can see the names of tables and columns it
--     was not granted. It cannot read their rows.
--   * Network access (pg_hba.conf, firewall, TLS) is not configured here.

\set ON_ERROR_STOP on

\if :{?pilot_role}
\else
  \set pilot_role mlpilot_pilot
\endif
\if :{?pilot_schemas}
\else
  \set pilot_schemas public
\endif
\if :{?pilot_statement_timeout}
\else
  \set pilot_statement_timeout 60s
\endif
\if :{?pilot_idle_timeout}
\else
  \set pilot_idle_timeout 60s
\endif
\if :{?pilot_connection_limit}
\else
  \set pilot_connection_limit 4
\endif
\if :{?revoke}
\else
  \set revoke no
\endif

SELECT :'revoke' = 'no' AS pilot_create,
       :'revoke' = 'lock' AND EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'pilot_role') AS pilot_lock,
       :'revoke' = 'drop' AND EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'pilot_role') AS pilot_drop
\gset

\if :pilot_create

\if :{?pilot_tables}
\else
  \echo 'readonly_role.sql: set -v pilot_tables=... (the tables MLPilot may read, schema-qualified)'
  DO $$ BEGIN RAISE EXCEPTION 'pilot_tables is not set'; END $$;
\endif

\if :{?pilot_password}
  \set pilot_password_clause 'PASSWORD ':'pilot_password'
\else
  \echo 'readonly_role.sql: pilot_password is not set; the role gets no password (use \password)'
  \set pilot_password_clause ''
\endif

BEGIN;

SELECT count(*) = 0 AS pilot_role_missing FROM pg_roles WHERE rolname = :'pilot_role' \gset
\if :pilot_role_missing
  CREATE ROLE :"pilot_role" LOGIN;
\endif

-- The role can log in and do nothing else.
ALTER ROLE :"pilot_role" WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    NOBYPASSRLS NOINHERIT CONNECTION LIMIT :pilot_connection_limit :pilot_password_clause;

-- Defaults for every new session of this role.
ALTER ROLE :"pilot_role" SET default_transaction_read_only = on;
ALTER ROLE :"pilot_role" SET statement_timeout = :'pilot_statement_timeout';
ALTER ROLE :"pilot_role" SET idle_in_transaction_session_timeout = :'pilot_idle_timeout';

-- Start from nothing in the chosen schemas, then grant exactly the list.
REVOKE ALL ON ALL TABLES IN SCHEMA :pilot_schemas FROM :"pilot_role";
GRANT CONNECT ON DATABASE :"DBNAME" TO :"pilot_role";
GRANT USAGE ON SCHEMA :pilot_schemas TO :"pilot_role";
GRANT SELECT ON TABLE :pilot_tables TO :"pilot_role";

-- To keep a sensitive column away from MLPilot, grant columns instead of the whole table:
--   REVOKE SELECT ON public.customers FROM mlpilot_pilot;
--   GRANT SELECT (customer_id, signup_at, country) ON public.customers TO mlpilot_pilot;

COMMIT;

-- What the role has now.
SELECT rolname, rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolconnlimit, rolconfig
FROM pg_roles WHERE rolname = :'pilot_role';
SELECT table_schema, table_name, string_agg(privilege_type, ', ') AS privileges
FROM information_schema.role_table_grants WHERE grantee = :'pilot_role'
GROUP BY table_schema, table_name ORDER BY 1, 2;

\endif

-- Revoking access.
--
--   Stop it now, keep the role (reversible: run the script again):
--     psql ... -v revoke=lock -f readonly_role.sql
--   Remove it (grants in THIS database, then the role). Repeat in every database where you
--   granted it something, because DROP ROLE fails while it still holds privileges anywhere:
--     psql ... -v revoke=drop -f readonly_role.sql
--   Both do nothing if the role does not exist.
--
-- By hand, the same thing is:
--   ALTER ROLE mlpilot_pilot NOLOGIN;
--   SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = 'mlpilot_pilot';
--   DROP OWNED BY mlpilot_pilot;
--   DROP ROLE mlpilot_pilot;
-- Changing the password alone does not end sessions that are already open.

\if :pilot_lock
  ALTER ROLE :"pilot_role" NOLOGIN;
  SELECT pg_terminate_backend(pid) FROM pg_stat_activity
  WHERE usename = :'pilot_role' AND pid <> pg_backend_pid();
\endif

\if :pilot_drop
  ALTER ROLE :"pilot_role" NOLOGIN;
  SELECT pg_terminate_backend(pid) FROM pg_stat_activity
  WHERE usename = :'pilot_role' AND pid <> pg_backend_pid();
  DROP OWNED BY :"pilot_role";
  DROP ROLE :"pilot_role";
\endif

-- MySQL / MariaDB (notes only: MLPilot connects to PostgreSQL, SQLite and DuckDB today, so this
-- has not been run against anything and no test covers it). The same idea in MySQL 8:
--
--   CREATE USER 'mlpilot_pilot'@'10.0.0.%' IDENTIFIED BY '...' WITH MAX_USER_CONNECTIONS 4;
--   GRANT SELECT ON shop.customers TO 'mlpilot_pilot'@'10.0.0.%';   -- one statement per table
--   -- or per column: GRANT SELECT (customer_id, signup_at) ON shop.customers TO ...
--
-- MySQL has no per-user default for read-only transactions or for the statement timeout
-- (max_execution_time is a server or session setting, or the MAX_EXECUTION_TIME(ms) optimizer
-- hint on a SELECT), so the grant is the only per-user control. Revoke with
--   DROP USER 'mlpilot_pilot'@'10.0.0.%';
-- which also ends no open sessions: run KILL for those. Replicas are a good place to point a
-- pilot at, in either database.
