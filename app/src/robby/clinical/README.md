# clinical

Recording admissions, classifying them in the Robson system, scoring the readiness signal and
recording the delivery outcome.

- `form.py`: reads the admission form (`parse`). Nothing is coerced: each field is whole text
  that fits it, or an error naming the field; a blank is "not entered", never zero. Each
  screening test needs a value or "not measured" with a reason.
- `importer.py`: Excel and JSON batches. A `SheetLayout` (template, `openclinic.py`,
  `research_export.py`) translates the sheet into the template layout, then every row goes
  through `form.parse`. `find_duplicates` spots repeated patient number + admission time
  (compared by blind index, `robby.crypto`).
- `api.py`: `POST /api/v1/admissions/import` for scripts (bearer token, all-or-nothing).
- `classify.py`: classification against the active rule set (`robson_engine`) and the panel
  view shared by the live form and the result page.
- `scoring.py`: `ModelRuntime`, a per-process cache of loaded models keyed by (model, rule set),
  with a `ScorerLoader`/`Scorer` protocol so tests can serve a stub; a model that fails to load
  is retried after a pause and never blocks saving. `score_row`, `yes_no_level` and `with_input`
  re-score a stored feature vector with one input filled in, spelled by the model itself (used by
  `robby.lab` after an HIV result).
- `service.py`: one transaction per save (admission, omissions, classification, active and
  shadow predictions, event); reads batched per list (`rows_of`, `summaries_of`); outcomes.
  `prediction_of` and the batched percent read the current prediction: the one no re-score
  supersedes.
- `views.py`: the clinician's result page (`ResultView`, including when the signal was updated
  after an HIV result).
- `routes.py`: the lab technician's form, the clinician's patient search, result and outcome
  pages, and the administrator's Excel import.
- `labels.py`, `xlsx_template.py`: the words and options shown, and the downloadable template.

Clinical rows are never updated: a correction is a new admission whose `supersedes_id` names
the original, and outcomes follow that chain.

Lab requests live next door in `robby/lab/`: the clinician requests tests from the result page,
the lab technician records results on `/lab-requests`. The result page lists the requests and
shows screening tests completed later; an HIV positive or negative result re-scores the signal
as a new prediction that supersedes the current one (`lab/service.py`, `_rescore`). See
[Lab requests](../../../README.md#lab-requests) in the app README.
