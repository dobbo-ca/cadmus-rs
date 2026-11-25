# Benchmark

`bench.yml` (dispatch and push to `main`, not required) runs `oracle -bench` (Go) and
`examples/bench.rs` (Rust) in one container on the same model, the line corpus and the scan, and
prints the ratio per class in the job summary (`tests/parity/bench.sh`) three times: with both at their
default parallelism (Go's default `GOMAXPROCS`, Rust's default `max_threads` on a caller-held
`ThreadPool` reused across calls, as a server would), with both on one thread (`GOMAXPROCS=1`,
`max_threads` 1), and Go at its default against Rust with no caller pool (the crate's shared pool).
Every class must be at or below 1.10x Go in all three tables.

`bench.yml` also prints two v4 tables, a report and not a gate: the Go oracle built with
`GOAMD64=v4` against the bench example built with `-C target-cpu=x86-64-v4` and the false-deps
features ([performance.md](performance.md)), at the default parallelism and on one thread. The v4
tables are printed only on a runner whose CPU has AVX-512.

## Hosted runner baseline

Bench run 45709439481 (2026-11-25) on a GitHub-hosted `ubuntu-24.04` runner, 2 CPUs visible to
the container, AVX-512 present (the v4 tables ran). It is the baseline for later hosted runs; do
not compare it with the Ryzen 7 9800X3D numbers below as a regression. Rust / Go per class, default
parallelism and one thread:

| class | default | 1 thread | v4 1 thread |
|---|---|---|---|
| recognize_native | 0.85x | 0.87x | 0.87x |
| forward_native | 0.85x | 0.86x | 0.87x |
| beam_native | 0.67x | 0.68x | 0.67x |
| skew_angle | 0.45x | 0.51x | 0.42x |
| scale_1.7 | 0.64x | 0.66x | 0.64x |
| sauvola_15 | 0.99x | 1.11x | 1.12x |

Every other class is between 0.32x and 0.78x. On this runner `sauvola_15` on one thread is above
the 1.10x limit (1.11x; 1.05x with no caller pool), so the run fails the gate; that class is an
open item. The v4 build gains on `skew_angle` (one thread 29.3 ms to 23.8 ms) and is within the
noise elsewhere; the recognizer classes are about 5% slower with v4.

## PGO and BOLT

`pgo.yml` on an AMD Ryzen 7 9800X3D, Rust ms per class, the
change against the default build ([performance.md](performance.md) says how it trains):

| class | default threads: default | PGO | PGO + BOLT | 1 thread: default | PGO | PGO + BOLT |
|---|---|---|---|---|---|---|
| forward_native | 1727.2 | 1693.2 (-2.0%) | 1695.0 (-1.9%) | 1791.5 | 1773.3 (-1.0%) | 1768.9 (-1.3%) |
| forward_h36 | 1726.6 | 1689.6 (-2.1%) | 1700.1 (-1.5%) | 1784.7 | 1772.6 (-0.7%) | 1782.8 (-0.1%) |
| recognize_native | 1736.8 | 1698.9 (-2.2%) | 1704.1 (-1.9%) | 1797.5 | 1782.1 (-0.9%) | 1782.7 (-0.8%) |
| recognize_h36 | 1731.3 | 1696.3 (-2.0%) | 1708.8 (-1.3%) | 1791.5 | 1776.2 (-0.8%) | 1772.4 (-1.1%) |
| beam_native | 6.99 | 6.77 (-3.2%) | 6.64 (-5.0%) | 7.02 | 6.82 (-2.9%) | 6.49 (-7.6%) |
| skew_angle | 3.56 | 3.18 (-10.7%) | 3.00 (-15.9%) | 15.55 | 17.08 (+9.8%) | 15.22 (-2.1%) |
| scale_1.7 | 1.87 | 1.92 (+2.7%) | 1.43 (-23.4%) | 10.14 | 11.14 (+9.9%) | 11.18 (+10.3%) |
| rotate | 0.381 | 0.394 (+3.4%) | 0.371 (-2.6%) | 1.90 | 2.52 (+32.7%) | 2.49 (+31.1%) |
| otsu | 0.173 | 0.207 (+19.7%) | 0.202 (+16.8%) | 1.008 | 1.110 (+10.1%) | 0.974 (-3.4%) |

The other imaging classes change by -14% to +13% (`scale_0.5` +12.7% with PGO at default threads); the full tables are in the run's job summary.

Result: for a recognizer worker PGO (with or without BOLT) gains about 1 to 2%, close to the
runner noise; the LSTM matrix loop is the limit and the profile does not change it. On the imaging
operators it is mixed: the morphology and beam classes gain up to 14%, but `otsu` (+20% at default
threads) and `scale_0.5` (+13%) are slower, and single-threaded `rotate`
is 31% and `scale_1.7` 10% slower. Do not ship a profile trained on this bench; a worker that
wants PGO must train on its own line stream and check the imaging calls it uses against the default
build.
