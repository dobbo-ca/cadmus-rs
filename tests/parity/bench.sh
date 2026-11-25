#!/bin/sh
# Times the Go library (tests/parity/go/oracle -bench) and this crate (examples/bench.rs) on the same
# model, line corpus and scan, in one container, and prints three markdown tables with the ratio:
# both at their default parallelism (Go's default GOMAXPROCS, Rust's default
# `Limits::max_threads` on a caller-held `ThreadPool` reused across calls, as a server would,
# like Go's warm goroutine runtime), both on one thread (GOMAXPROCS=1, `max_threads` = 1), and
# Go at its default against Rust with no caller pool (`Limits::default()`: the crate's lazy
# shared pool). Every class must be at or below 1.10x Go in all three (conventions.md): the script
# exits non-zero if one is not, after printing all three tables.
# The Rust binary is $BENCH (default `bench`). When `oracle-v4` (Go built with GOAMD64=v4) and
# $BENCH_V4 (default `bench-v4`, Rust built with -C target-cpu=x86-64-v4 and the false-deps features
# of docs/performance.md) are on PATH, two more tables compare the v4 builds (default parallelism and
# one thread); they are a report, not a gate (the 1.10x rule is on the default builds). They run only
# on a CPU with AVX-512 (/proc/cpuinfo lists avx512f): a v4 binary dies with SIGILL elsewhere.
set -e
rc=0
bench=${BENCH:-bench}
bench4=${BENCH_V4:-bench-v4}
oracle=oracle
gate=1
table() { # title, Go command prefix, Rust bench arguments
	title=$1 go=$2 rs=$3
	$go $oracle -bench -model tests/data/eng.traineddata tests/data | sort > /tmp/go.tsv
	$bench tests/data/eng.traineddata tests/data $rs | sort > /tmp/rs.tsv
	# A class only one side prints would be left out of the join: fail instead.
	cut -f1 /tmp/go.tsv > /tmp/go.names
	cut -f1 /tmp/rs.tsv > /tmp/rs.names
	cmp -s /tmp/go.names /tmp/rs.names || { echo 'bench classes differ between Go and Rust:' >&2; diff /tmp/go.names /tmp/rs.names >&2; exit 1; }
	echo "#### $title"
	echo
	echo '| class | Go ms | Rust ms | Rust / Go |'
	echo '|---|---|---|---|'
	# conventions.md: Rust at or below 1.10x Go on every class; a class above it fails the script.
	join -t "$(printf '\t')" /tmp/go.tsv /tmp/rs.tsv | awk -F'\t' -v gate="$gate" '
{ printf "| %s | %.3f | %.3f | %.2fx |\n", $1, $2, $3, $3 / $2 }
gate && $3 / $2 > 1.10 { printf "::warning::%s is %.2fx Go (limit 1.10)\n", $1, $3 / $2 > "/dev/stderr"; bad = 1 }
END { exit bad }' || rc=1
	echo
}
echo "$(nproc) CPUs visible to the container"
echo
table "Default parallelism: Go default GOMAXPROCS, Rust default max_threads" "env" ""
table "One thread: Go GOMAXPROCS=1, Rust max_threads=1" "env GOMAXPROCS=1" "1"
table "No caller pool: Go default GOMAXPROCS, Rust default max_threads on the shared pool" "env" "nopool"
if ! grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
	echo "no AVX-512 on this CPU: the v4 tables are skipped" >&2
elif command -v oracle-v4 >/dev/null 2>&1 && command -v "$bench4" >/dev/null 2>&1; then
	oracle=oracle-v4 bench=$bench4 gate=0
	table "v4 builds, default parallelism: Go GOAMD64=v4, Rust -C target-cpu=x86-64-v4 (report only)" "env" ""
	table "v4 builds, one thread: Go GOAMD64=v4 GOMAXPROCS=1, Rust x86-64-v4 max_threads=1 (report only)" "env GOMAXPROCS=1" "1"
fi
exit $rc
