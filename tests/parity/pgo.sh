#!/bin/sh
# Times the bench example built three ways: tests/parity/pgo.sh. Needs `oracle` (Go) and `bench` (the
# default build), `bench-pgo` (cargo-pgo, trained on the bench inputs) and, when BOLT was available,
# `bench-bolt` (PGO then BOLT) on PATH; run from the repo root (stage `pgo` of .github/ci/Dockerfile,
# workflow pgo.yml). Each build runs tests/parity/bench.sh (its Rust binary is $BENCH; its 1.10x gate
# does not fail this script); the tables give the Rust ms of each class, at the default threads and
# on one thread, and the change against the default build.
set -eu
export LC_ALL=C
t=$(mktemp -d)
have=
labels=
for v in bench:default bench-pgo:PGO bench-bolt:"PGO + BOLT"; do
	b=${v%%:*}
	if ! command -v "$b" >/dev/null 2>&1; then
		echo >&2 "no $b: skipped"
		continue
	fi
	BENCH=$b sh tests/parity/bench.sh >"$t/$b.md" || true
	have="$have $t/$b.md"
	labels="$labels,${v#*:}"
done
[ -n "$have" ] || { echo >&2 "no bench binary on PATH"; exit 1; }
# bench.sh prints a "#### title" line before each table; table 1 is the default parallelism, table 2
# one thread. A class row is `| name | Go ms | Rust ms | ratio |`.
# shellcheck disable=SC2086
awk -F'|' -v labels="${labels#,}" '
FNR == 1 { nf++; tab = 0 }
/^####/ { tab++ }
tab <= 2 && /^[|] [a-z0-9_.]+ [|] [0-9]/ {
  name = $2; gsub(/ /, "", name)
  v[nf, tab, name] = $4 + 0
  if (!(name in seen)) { seen[name] = 1; names[++cnt] = name }
}
END {
  split(labels, lab, ",")
  split("default threads,single-threaded", pass, ",")
  for (p = 1; p <= 2; p++) {
    print "#### Rust ms, " pass[p] " (lower is better; the change is against the default build)"
    print ""
    s = "| class |"; r = "|---|"
    for (f = 1; f <= nf; f++) { s = s " " lab[f] " |"; r = r "---|"; if (f > 1) { s = s " change |"; r = r "---|" } }
    print s; print r
    for (i = 1; i <= cnt; i++) {
      k = names[i]
      s = "| " k " |"
      for (f = 1; f <= nf; f++) {
        s = s sprintf(" %.3f |", v[f, p, k])
        if (f > 1 && v[1, p, k] > 0) s = s sprintf(" %+.1f%% |", (v[f, p, k] / v[1, p, k] - 1) * 100)
      }
      print s
    }
    print ""
  }
}' $have
for f in $have; do
	echo "<details><summary>$(basename "$f" .md): the full bench.sh tables</summary>"
	echo
	cat "$f"
	echo
	echo "</details>"
done
rm -rf "$t"
