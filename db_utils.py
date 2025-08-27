import os, socket, urllib.parse
import streamlit as st
from sqlalchemy import create_engine

@st.cache_resource
def get_engine():
    url = st.secrets.get("DB_URL") or os.environ.get("DB_URL")
    if not url:
        cfg = st.secrets["postgres"]  # requires [postgres] block in secrets
        enc_pwd = urllib.parse.quote_plus(cfg["password"])
        host = cfg["host"]
        port = cfg.get("port", 5432)
        db   = cfg["database"]
        user = cfg["user"]
        url = f"postgresql+psycopg2://{user}:{enc_pwd}@{host}:{port}/{db}?sslmode=require"

    # DNS sanity check (shows only host, no secrets)
    host_only = url.split("@", 1)[1].split("/", 1)[0].split(":")[0]
    socket.gethostbyname(host_only)

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=3,
        max_overflow=0,
    )
