# Robson Readiness web application

Full README arrives in a later task.

## Development database

A private PostgreSQL cluster (trust auth, 127.0.0.1 only) avoids touching any system install:

```bash
sh scripts/dev-db.sh init     # once: creates $ROBBY_PGDATA (default ~/robby-pgdata), role robby_dev, database robby
sh scripts/dev-db.sh start    # also: stop, status
```

Point `app/.env` at it (see `.env.example`), for example
`ROBBY_DATABASE_URL=postgresql+psycopg://robby_dev@127.0.0.1:55432/robby` and
`ROBBY_TEST_DATABASE_URL=postgresql+psycopg://robby_dev@127.0.0.1:55432/postgres`.
