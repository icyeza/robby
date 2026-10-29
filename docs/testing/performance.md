# Performance and compatibility measurements

Measured on 29 October 2026 against commit `e90e34a`; the operating-system runs against `89ddb1d`.

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

## Operating systems and Python versions

| Environment | Suite | Result |
|---|---|---|
| Windows 11 (laptop), Python 3.11, PostgreSQL 16 | app | 428 passed, 1 skipped |
| Ubuntu 24.04 (server), Python 3.11, PostgreSQL 16 | app | 428 passed, 1 skipped |
| Ubuntu 24.04, Python 3.11, 3.12 and 3.13 | `robson_engine`, `robson_serving` | 130 and 107 passed on each |
| Windows 11, Python 3.11 | `robson_engine`, `robson_serving`, ML pipeline | 130, 107 and 651 passed |
