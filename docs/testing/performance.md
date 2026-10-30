# Performance and compatibility measurements

Measured on 29 October 2026 against commit `c93f986`; the operating-system runs against `847680d` (packages) and `9fe1050` (app).

## Server performance (requirement NFR1)

**Set-up.** The application ran as deployed: Uvicorn with 2 workers behind systemd, and PostgreSQL 16
on the same machine. It used a separate demonstration database of 14,000 synthetic admissions
loaded through the real services (`robby-admin demo load --admissions 14000`). The busiest hospital
in it, Demo Hospital D, has **4,736 deliveries**. That is more than one year of births at any of
the four Kigali hospitals in the study data (5,520 deliveries across all four in five months).

**Method.** [`app/scripts/perf_check.py`](../../app/scripts/perf_check.py) was the client. It ran
on the server itself, so network time is excluded, and signed in as that hospital's demo accounts.

**Machine.** A VPS with 6 vCPUs (Intel Broadwell, 2.0 GHz) and 11 GB RAM, running Ubuntu 24.04.
The machine is shared with about 30 other services (load average about 5 during the runs). The two
smaller profiles were imposed on the same service with systemd resource limits (`CPUQuota`,
`MemoryMax`).

All times are in milliseconds.

### Full server (6 vCPUs, shared)

| Measure | Requests | p50 | p95 | max | Errors |
|---|---|---|---|---|---|
| Save an admission (parse, classify, score, store, log) | 50 | 149 | 191 | 198 | 0 |
| Open a saved admission | 50 | 24 | 29 | 30 | 0 |
| Hospital report, 4,736 deliveries | 20 | 172 | 193 | 193 | 0 |
| Hospital report PDF | 20 | 263 | 347 | 348 | 0 |
| 8 simultaneous users (form and report pages) | 1,680 | 90 | 399 | 779 | 0 |

Throughput was 55.5 requests per second, using 64 CPU-seconds over 50 s.

### One CPU core, 1 GB memory cap

| Measure | Requests | p50 | p95 | max | Errors |
|---|---|---|---|---|---|
| Save an admission | 50 | 166 | 210 | 248 | 0 |
| Open a saved admission | 50 | 23 | 53 | 109 | 0 |
| Hospital report, 4,736 deliveries | 20 | 188 | 316 | 320 | 0 |
| Hospital report PDF | 20 | 274 | 306 | 306 | 0 |
| 8 simultaneous users | 1,132 | 127 | 464 | 903 | 0 |

Throughput was 37.1 requests per second.

### Half a CPU core, 1 GB memory cap

| Measure | Requests | p50 | p95 | max | Errors |
|---|---|---|---|---|---|
| Save an admission | 50 | 307 | **529** | **991** | 0 |
| Open a saved admission | 50 | 32 | 77 | 79 | 0 |
| Hospital report, 4,736 deliveries | 20 | 176 | 222 | 223 | 0 |
| Hospital report PDF | 20 | 326 | 507 | 509 | 0 |
| 8 simultaneous users | 431 | 388 | 707 | 1,395 | 0 |

Throughput was 14.2 requests per second.

### Resources

- **Memory.** The two workers use about 400–440 MB at rest and peaked at **about 950 MB** under
  the 8-user load. Most of that is pandas and scikit-learn loaded in each worker. A 1 GB cap was
  never reached.
- **Cold start.** Loading the rule set and the model takes about 7 s per worker.
- **NFR1 verdict.**
  - Saving an admission stays under the 500 ms limit at one core or more.
  - The facility-year report stays under 3 s in every profile, at most 0.3 s.
  - At half a core, saving exceeds 500 ms at the 95th percentile. One core is therefore the
    minimum we state for a server.

## Browsers

[`app/scripts/browser_check.py`](../../app/scripts/browser_check.py) ran on a Windows 11 laptop
against the deployed demonstration server, through an SSH tunnel (Germany to the client). It goes
through the lab technician and clinician flow: sign in, fill the form, check the live Robson panel,
check that a blank screening blocks the save, save, find the patient, and read the result.

| Browser | Desktop 1366x768 | Phone 390x844 | Form load (incl. network) |
|---|---|---|---|
| Chrome 156 | pass | pass | 384–406 ms |
| Edge 156 | pass | pass | 395 ms |
| Firefox 157 | pass | pass | 506–590 ms |
| WebKit 27.2 (Safari's engine) | see note | see note | |

**WebKit note.** Over the plain-http tunnel, WebKit does not send the session cookie, which is
marked `Secure`. Chromium and Firefox treat `127.0.0.1` as a secure origin and do send it. On the
earlier local run (`ROBBY_SECURE_COOKIES=false`) WebKit passed at both sizes. The check is to be
repeated over HTTPS once the public address is live.

## Stress test (Locust): finding the breaking point

**Tool.** [`app/scripts/locustfile.py`](../../app/scripts/locustfile.py).

**Load.** Simulated staff of the four demo hospitals act in proportion:
- lab technicians open the form and save admissions (weight 3);
- clinicians search a patient by name and open the admission (weight 3);
- administrators open the whole-period hospital report (weight 1).

Each simulated user pauses 1–3 s between actions. The number of users rises in steps of one minute
each: 10, 25, 50, 100, 150 and 200.

**Target.** The deployed configuration (2 Uvicorn workers) on the 14,000-admission database, capped
at **2 CPU cores and 1.5 GB** with systemd, run on the server itself.

**First run** (commit `c44173e`; [report](stress/before.html)).
- Up to 45 users there were no errors, at a 95th percentile of 0.4–1.3 s.
- At 50 users the first errors appeared. At about 100 users the system collapsed: the median was
  33 s and throughput fell to 4 requests/s.
- All 127 failures were `QueuePool limit of size 5 overflow 10 reached … timeout 30`. CPU-bound
  requests held database connections, and the others waited 30 s before failing with a 500.
- Memory reached the 1.5 GB cap, mostly from bursts of sign-ins. Each argon2id hash uses 64 MiB.

**Changes** (`baf853a`).
- The connection pool is sized explicitly (10 + 10 per worker).
- A request that finds no free connection within 5 s gets a styled **503 "busy, try again"** with
  `Retry-After`, instead of a 500 after 30 s.
- At most two argon2id hashes run at once per process, keeping the RFC 9106 parameters.

**Second run** (commit `9fe1050`; [report](stress/after.html)).

| | First run | Second run |
|---|---|---|
| Requests served in the 6 minutes | 2,968 | **4,304** (+45%) |
| 10–45 users: p95, errors | 0.4–1.3 s, 0 | 0.5–1.1 s, 0 |
| 50–70 users: throughput, p95 | 13–17 req/s, 1.7–3 s | **20 req/s**, 3 s |
| 90 users: errors | 5 | **0** |
| 100 users: median, throughput | 33 s, 4 req/s | **5.1 s, 14 req/s** |
| 200 users: median | 60 s | 36 s |
| Failures | 127, all 500 after a 30 s wait | 187, all **503 "busy"** returned quickly |
| Server errors (500) | 127 | **0** |
| Memory peak | 1.57 GB (cap reached) | **0.90 GB** |

**Reading.**
- On 2 cores the system serves about **45 simultaneously active staff with p95 near 1 s**, and 90
  without a single error. That is far more than one maternity unit's staff on a shift.
- Beyond that it now degrades gracefully: it answers "busy" quickly, never fails with a server
  error, and stays within memory.
- Sign-in remains the slowest action under a burst, because hashes are queued two at a time.
  That is deliberate: real sign-ins are spread across a shift, and the hashing cost is what
  protects the passwords.

## Accessibility (axe-core)

[`app/scripts/a11y_check.py`](../../app/scripts/a11y_check.py) runs the WCAG 2.1 A and AA rules of
axe-core 4.10.2 on 18 screens covering every role. axe runs through the browser's debugging
protocol, so the strict CSP stays on.

- **First run:** one rule failed, `color-contrast`, on 8 elements across 4 screens: muted labels
  at 4.26:1, unresolved group numbers at 3.52:1 and the sign-in strip at 3.43:1.
- **Fix:** three colours changed (`bba185a`), each to at least 4.5:1.
- **Second run:** **no violations**.

## Dependencies (pip-audit)

pip-audit checked both lock files against the OSV database.

- **App:** no known vulnerabilities.
- **ML workspace, first check:** 11 packages with known vulnerabilities, all on the training and
  notebook side, none of them installed with the app: pillow, setuptools, tornado, werkzeug,
  aiohttp, certifi, filelock, gitpython, mlflow, nbconvert and opentelemetry-api.
- **Fix:** they were upgraded (`3dbf409`). The API, serving-parity and tracking tests pass, and the
  audit finds none.
- **CI:** both audits now run on every push.

## Operating systems and Python versions

| Environment | Suite | Result |
|---|---|---|
| Windows 11 (laptop), Python 3.11, PostgreSQL 16 | app | 628 passed, 1 skipped |
| Ubuntu 24.04 (server), Python 3.11, PostgreSQL 16 | app | 628 passed, 1 skipped |
| Ubuntu 24.04, Python 3.11, 3.12 and 3.13 | `robson_engine`, `robson_serving` | 130 and 107 passed on each |
| Windows 11, Python 3.11 | `robson_engine`, `robson_serving`, ML pipeline | 130, 107 and 651 passed |
