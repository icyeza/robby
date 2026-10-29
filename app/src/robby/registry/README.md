# registry

Model governance: which packaged model scores admissions, and the recorded approvals that change
it.

- `transitions.py`: the lifecycle candidate -> shadow -> active -> retired (and rollback of a
  retired model). Each move needs the custodian's own username and a written reason, locks the
  rows involved, demotes the current active model before promoting, and writes a
  model_transition row and one event. A partial unique index keeps exactly one active model.
- `evidence.py`: shadow against active on reconciled outcomes over `WINDOW_DAYS`: discrimination
  and calibration with bootstrap intervals, per-stratum rows, and a calibration plot.
- `approval_log.py`: a read-only view of approvals, corrections and account changes from the
  event log, and "Verify the log" (recomputes the hash chain).
- `service.py`: read queries for the pages (models, history, usernames).
- `routes.py`: models list and detail with the approval dialogs.
- `dev_register.py`: `robby-admin model register`, which validates a package (load and golden
  check), copies it under ARTEFACT_DIR and registers it as a candidate; `--activate` is a
  development shortcut that skips the approval flow and says so in the log.
