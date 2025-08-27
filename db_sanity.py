# Minimal CLI probe to test DB connectivity with the same logic as the app.
import sys, pandas as pd
from sqlalchemy import text
import streamlit as st  # used by get_engine for secrets
from db_utils import get_engine

def main():
    try:
        eng = get_engine()
    except Exception as e:
        print(f"[FAIL] Engine init: {e}")
        sys.exit(2)

    try:
        with eng.connect() as c:
            ok = c.execute(text("SELECT 1")).scalar()
            print(f"[OK] SELECT 1 => {ok}")
            tables = pd.read_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;",
                c
            )
            print("[INFO] Public tables:")
            for t in tables["table_name"].tolist():
                print(" -", t)
    except Exception as e:
        print(f"[FAIL] Query: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
