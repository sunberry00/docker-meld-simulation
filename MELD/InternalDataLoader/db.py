import os

from sqlalchemy import create_engine

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
schema = os.getenv("DB_SCHEMA")

# Support both Docker Secrets (DB_PASSWORD_FILE) and plain env (DB_PASSWORD).
password_file = os.getenv("DB_PASSWORD_FILE")
password = os.getenv("DB_PASSWORD")
if password_file and os.path.isfile(password_file):
    with open(password_file, "r") as f:
        password = f.readline().strip()

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{schema}")
