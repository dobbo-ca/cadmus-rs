#!/bin/sh
# Real-file OCR gate (docs/quality.md), run by the `quality` stage of .github/ci/Dockerfile:
#   tests/quality/gate.sh <rust-oracle> <model> <report-dir> <slide>...
# `labels` (tests/quality/labels, tiff-rs) cuts the text-line crops of the slides' label, macro and
# thumbnail pages once into /crops; when that gives fewer than MIN_CROPS, the line corpus of the
# parity job (tests/data/lines) is added. The Go oracle (`oracle` on PATH) and the Rust oracle then
# run the same crops (forward pass, greedy and beam decodes, recognize: text, words, confidences);
# the Rust one twice, with 1 thread and with the default threads, and, when $RS_V4 names it, once more
# built for x86-64-v4 (docs/performance.md) with the default threads. Every output tree must equal
# the Go one byte for byte. Writes summary.md, labels.log, the oracle logs, diff-*.txt and status (0
# equal, 1 a difference, 2 a run failed) into the report dir; the exit status is always 0, so the
# workflow can upload the report before it fails the job.
set -u
export LC_ALL=C
MIN_CROPS=100
rs=$1 model=$2 report=$3
shift 3
mkdir -p "$report"
status=0
fail() { echo "$1" >>"$report/errors.txt"; status=2; }

rm -rf /crops /go-out /rs-out-1 /rs-out-0 /rs-out-v4
mkdir -p /crops
labels /crops "$@" >"$report/labels.log" || fail "labels failed"
nlabel=$(ls /crops | wc -l)
[ "$nlabel" -ge 10 ] || fail "only $nlabel crops from the slides (no page was read or cut?)"
ncorpus=0
if [ "$nlabel" -lt "$MIN_CROPS" ]; then
	for s in native h36; do
		for f in tests/data/lines/$s/*.pgm; do
			cp "$f" "/crops/line-corpus-$s-$(basename "$f")"
			ncorpus=$((ncorpus + 1))
		done
	done
fi
ncrops=$(ls /crops | wc -l)

oracle -model "$model" /crops /go-out 2>"$report/go.log" || fail "Go oracle failed"
ngo=$(ls /go-out 2>/dev/null | wc -l)
[ "$ngo" -eq "$ncrops" ] || fail "Go wrote $ngo outputs for $ncrops crops"
for t in 1 0; do
	"$rs" -threads "$t" -model "$model" /crops "/rs-out-$t" 2>"$report/rs-$t.log" || fail "Rust oracle (-threads $t) failed"
	if ! diff -r /go-out "/rs-out-$t" >"$report/diff-$t.txt"; then
		[ "$status" -eq 2 ] || status=1
	fi
done
if [ -n "${RS_V4:-}" ]; then
	"$RS_V4" -threads 0 -model "$model" /crops /rs-out-v4 2>"$report/rs-v4.log" || fail "Rust v4 oracle failed"
	if ! diff -r /go-out /rs-out-v4 >"$report/diff-v4.txt"; then
		[ "$status" -eq 2 ] || status=1
	fi
fi

{
	echo "## OCR quality gate (exact): label line crops, Go cadmus vs cadmus-rs"
	echo
	echo "- Slides: $(for s in "$@"; do basename "$s"; done | tr '\n' ' ')"
	echo "- Pages cut (untiled, 8-bit, at most 8 Mpixel; crops per orientation):"
	echo
	grep ' spp ' "$report/labels.log" | sed 's/^/  - /'
	echo
	echo "- Crops: $nlabel from the slides$([ "$ncorpus" -gt 0 ] && echo ", plus $ncorpus lines of the parity corpus (tests/data/lines), because the slides gave fewer than $MIN_CROPS")"
	echo "- Outputs: Go $ngo; Rust 1 thread $(ls /rs-out-1 2>/dev/null | wc -l), default threads $(ls /rs-out-0 2>/dev/null | wc -l)$([ -n "${RS_V4:-}" ] && echo ", x86-64-v4 build $(ls /rs-out-v4 2>/dev/null | wc -l)")"
	echo "- Differing outputs: 1 thread $(grep -c '^diff \|^Only in' "$report/diff-1.txt"), default threads $(grep -c '^diff \|^Only in' "$report/diff-0.txt")$([ -n "${RS_V4:-}" ] && echo ", x86-64-v4 build $(grep -c '^diff \|^Only in' "$report/diff-v4.txt")")"
	if [ -s "$report/errors.txt" ]; then
		echo "- Errors:"
		sed 's/^/  - /' "$report/errors.txt"
	fi
	echo "- Verdict: $(case $status in 0) echo 'PASS: every output byte-identical (forward hash, greedy, beam, text, words, confidences)' ;; 1) echo 'FAIL: outputs differ' ;; *) echo 'FAIL: a run failed' ;; esac)"
	echo
	echo "### Recognized text per crop (Go output; the Rust output is the same when the verdict is PASS)"
	echo
	perl -e '
		my ($go, $crops) = @ARGV;
		opendir(my $d, $crops) or die;
		my @f = sort grep { /\.pgm$/ } readdir($d);
		my ($n, $sum, $nonempty) = (0, 0, 0);
		print "| crop | size | text | confidence |\n|---|---|---|---|\n";
		for my $f (@f) {
			open(my $p, "<", "$crops/$f") or die; binmode($p); read($p, my $hdr, 64);
			my ($w, $h) = $hdr =~ /^P5\s+(\d+)\s+(\d+)/;
			my ($t, $c) = ("(no output)", "");
			if (open(my $o, "<", "$go/$f.out")) {
				while (<$o>) {
					if (/^recognize: text=([0-9a-f]*) conf=([0-9a-f]{16})/) {
						$t = pack("H*", $1);
						my $v = unpack("d>", pack("H16", $2));
						$c = sprintf("%.3f", $v); $n++; $sum += $v; $nonempty++ if length($t);
					} elsif (/^recognize: ERR (.*)/) { $t = "ERR $1"; }
				}
			} elsif (-e "$go/$f.err") { $t = "ERR (input)"; }
			$t =~ s/[\x00-\x1f`|]/ /g;
			print "| $f | ${w}x$h | `$t` | $c |\n";
		}
		printf "\n%d crops recognized, %d with text, mean confidence %.3f\n", $n, $nonempty, $n ? $sum / $n : 0;
	' /go-out /crops
} >"$report/summary.md"
echo "$status" >"$report/status"
rm -rf /crops /go-out /rs-out-1 /rs-out-0 /rs-out-v4
exit 0
