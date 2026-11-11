from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

# Use native PostgreSQL JSONB in production while staying compatible with SQLite in unit tests.
JSONType = JSONB().with_variant(JSON(), "sqlite")

