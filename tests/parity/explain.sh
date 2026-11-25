#!/bin/sh
# Run by the parity stage when the trees differ: for each .mut input in /tmp/diff.txt, the recipe,
# both sides' outputs and, from a rerun of each oracle on that input alone, their error messages.
for f in $(grep -o 'mut-[A-Za-z0-9-]*\.mut' /tmp/diff.txt | sort -u); do
  echo "=== $f"; cat /inputs/$f
  rm -rf /one /one-go /one-rs; mkdir /one; cp /inputs/$f /one/
  ORACLE_MSG=1 oracle -model /model/eng.traineddata /one /one-go 2>&1 | head -5
  ORACLE_MSG=1 /src/target/release/examples/oracle -model /model/eng.traineddata /one /one-rs 2>&1 | head -5
  for d in /one-go /one-rs; do echo "-- $d"; head -c 700 $d/$f.*; done
done
