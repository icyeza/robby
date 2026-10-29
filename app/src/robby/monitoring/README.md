# monitoring

How the active model behaves after deployment (FR12), for the model custodian.

- `reconcile.py`: the reconciliation job. It joins every prediction (active and shadow) to the
  delivery outcome of its admission's chain, through the chain root, inserting one row per
  prediction and never updating; a Postgres advisory lock keeps concurrent runs apart.
- `service.py`: the monitoring page, computed on each request from predictions, reconciled
  outcomes and the event log (there is no alerts table). Over the last `SCORED_DAYS` days:
  - outcome recording per facility, for admissions older than `OUTCOME_GRACE`;
  - calibration-in-the-large overall and observed against predicted rate per facility
    (rolling `CALIBRATION_DAYS` window);
  - input drift (population stability index) and missing-rate shifts against the training
    reference distributions.
  Each problem is an alert keyed by kind, subject and ISO week; acknowledging one writes an
  event, so the acknowledgement lasts for that week.
- `readiness.py`: the "high readiness" flag from each facility's readiness cut.
- `routes.py`: the monitoring page and alert acknowledgement.

The statistics themselves come from `robson_serving.monitoring`.
