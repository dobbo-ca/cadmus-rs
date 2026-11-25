# Real-file quality gate

Rule: on real files the output of
this crate must be byte-identical to the Go library's. `quality.yml` runs on push to `main`, on
dispatch, and on a pull request that carries the label `quality` (when the label is added and on
every push after that); it is not a required check, and it never cancels a run that is in progress.

```
 OpenSlide public test slides (quality-data stage of .github/ci/Dockerfile)
        |
 labels /crops <slides>        tests/quality/labels: tiff-rs + jpeg-rs decode, line cut, once
        |               \
 Go oracle /crops /go-out   examples/oracle.rs: 1 thread, default threads, x86-64-v4 build
        \               /
 diff -r, tests/quality/gate.sh  ->  summary.md, labels.log, diff-*.txt, status
```

- Inputs: the ten TIFF slides of the subset (Aperio SVS JPEG and JPEG 2000, Hamamatsu NDPI, Leica
  SCN, generic TIFF, Philips TIFF, Ventana BIF; the MIRAX zip is not a TIFF). OCR needs text, so
  the gate reads the label, macro and thumbnail pages, not the tiles: every page that is not tiled,
  8-bit grey or RGB, and at most 8 Mpixel.
- `tests/quality/labels` (its own manifest, a pinned [tiff-rs](https://github.com/dobbo-ca/tiff-rs)
  rev) decodes each such page to grey and cuts it in three orientations (as stored,
  90 and 270 degrees): an Otsu threshold, then every run of rows with ink that is 8 to 160 rows
  high becomes one crop (ink columns plus a 4 pixel margin, at most 40 per page and orientation; a
  taller run, where every row of a photographed area has a few ink pixels, is cut again with a row
  threshold of 3% of the page width),
  `line-<slide>-p<page>-r<deg>-<n>.pgm`. The crops are cut once and both sides read the same
  files, so the cutter does not have to match either library.
- Fewer than 10 crops from the slides fails the gate (status 2: no page was read or cut). The
  Aperio label pages proper (387x463 and 415x422, LZW) give no crop: they hold a barcode and
  little text; the crops come from the macro, thumbnail and Ventana/Philips pages.
- If the slides give fewer than 100 crops, the 30 line images of the parity corpus (`tests/data/lines`)
  are added, and the summary says so.
- Each crop runs through the full recognizer report of the parity oracles (`line-` inputs): the
  imaging operators, the forward pass (hash of every activation), the greedy decode, the beam
  search with and without the lexicon, and `recognize` (text, words, bounds, confidences as f64
  bits). The Rust oracle runs three times: 1 thread, the default threads, and built with the
  x86-64-v4 flags of [performance.md](performance.md) (only on a CPU with AVX-512). Each tree must
  equal the Go tree.
- The report (artifact `quality-ocr`): `summary.md` (also in the job summary) lists the pages cut,
  the crop counts, the differing outputs per Rust run, the verdict, and the recognized text and
  confidence of every crop; `labels.log`, `go.log`, `rs-*.log`, `diff-*.txt`. `status` is 0 when all
  outputs are equal, 1 when one differs, 2 when a run fails; the job exits with it.
