CREATE DATABASE prefect;
CREATE DATABASE mlflow;

DO $$ BEGIN
  CREATE USER prefect_user WITH PASSWORD 'prefect1234';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE USER mlflow_user WITH PASSWORD 'mlflow1234';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

GRANT ALL PRIVILEGES ON DATABASE prefect TO prefect_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;

-- Grant schema permissions (required in PostgreSQL 15+)
\c mlflow
GRANT ALL ON SCHEMA public TO mlflow_user;

\c prefect
GRANT ALL ON SCHEMA public TO prefect_user;