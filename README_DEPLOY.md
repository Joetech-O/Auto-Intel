# Streamlit + Render Postgres Deployment (Auto-Intel)

## Secrets (Streamlit Cloud)
Set one of:
```toml
DB_URL = "postgresql+psycopg2://USER:ENCODED_PASSWORD@dpg-abc123.region-postgres.render.com:5432/DBNAME?sslmode=require"
```
or
```toml
[postgres]
host     = "dpg-abc123.region-postgres.render.com"
port     = 5432
database = "auto_intel"
user     = "auto_intel_ro"
password = "CHANGE_ME"
```

## Code usage
```python
from db_utils import get_engine
import pandas as pd
eng = get_engine()
df = pd.read_sql("SELECT * FROM newcar_reviews;", eng)  # adjust table name as needed
```

## Troubleshooting
- Hostname must include region and `-postgres.render.com`
- Append `?sslmode=require` to the URL
- Passwords with special chars must be URL-encoded
- Use `db_sanity.py` to list public tables
