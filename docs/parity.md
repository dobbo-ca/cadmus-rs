# Parity with Go

This page has the inputs, the report formats and the error-class mapping. The `parity`
job runs the Go library (a snapshot of the sources at b7fac51 in `tests/parity/go/upstream`) and this
crate over the same inputs: the line corpus and the Leptonica scan with mutations, 400 random
bitmaps, 150 activation vectors, and edited copies of the model and of each of its components; it
compares the imaging operators, the recognizer's output down to the bits of every float, the
parsers' summaries and their errors, and fails on any difference in `diff -r`. The Rust side runs
four times, with `-threads 1`, with every CPU on the crate's shared pool, on a caller's pool
(`-pool caller`) and on per-call workers (`-pool percall`), and every output must equal Go's;
`parity-arm64` runs
a subset of the inputs on the oracle built for aarch64 and run natively on an arm64 runner. The claim is parity
modulo the rewrites listed on this page (Go's non-UTF-8 unicharset, out-of-memory and
runtime-panic outcomes are mapped onto the crate's errors); the go-run log counts how often each
fires.

`tests/parity/go/upstream/` is the non-test source of dobbo-ca/cadmus at b7fac51 (`internal/`, `cmd/cadmusdump`,
`go.mod`, `LICENSE`, `NOTICE`). Two deviations from the snapshot: the `go` directive of `go.mod` is
1.27.1 (the repo says 1.27.3), because the CI image runs with `GOTOOLCHAIN=local`; and
`internal/nn/gen` (the tables generator, `//go:build ignore`) is not copied. `tests/parity/go/oracle/` is a Go
program whose module path (`github.com/dobbo-ca/cadmus/oracle`) sits below cadmus's, so Go's
`internal` rule lets it import `cadmus/internal/...`; its `go.mod` `replace`s cadmus with
`../upstream`.

The CI job `parity` (`.github/ci/Dockerfile`, stages `model`, `go-oracle`, `go-run`, `parity`) runs:

1. `oracle -model eng.traineddata -gen tests/data /inputs` writes the inputs.
2. `oracle -model eng.traineddata /inputs /go-out` runs them with the Go library.
3. `oracle -threads 1 -model eng.traineddata /inputs /rust-out` (the `--release --example oracle`
   binary) runs them with this crate on one thread, and then `-threads 0` (the default,
   `max_threads` = every CPU, on the crate's shared pool) runs them again, then `-threads 0 -pool
   caller` (one `ThreadPool` of the caller's, reused for every input) and `-threads 0 -pool
   percall` (workers the call builds for itself, as when the shared pool cannot be built).
4. `diff -r /go-out /rust-out` must be empty for each of the four runs (and `/go-out` must hold at
   least 1000 files, so an empty diff cannot mean an empty run). A difference is a bug in the Rust
   port, never in the oracle.
5. `cadmusdump`: the Go tool (built from `tests/parity/go/upstream/cmd/cadmusdump`) and `cargo run --release
   --bin cadmusdump` run on the same model path with no flag, `-unicharset`, `-recoder`, `-dawgs`,
   all three, and `-activations` on `tests/data/lines/h36/0001.pgm` (the Go tool reads a PNG: `oracle
   -topng` writes it from the PGM). The six outputs must be byte for byte equal (`diff -r
   /go-dump /rust-dump`).

The CI job `parity-arm64` (stages `go-run-arm64`, `go-arm64-data`, `parity-arm64`) runs a subset of the inputs
(the 150 `vec-*.nn` vectors, `golden-gray.pgm`, `rnd-0000` to `rnd-0009`, the first two `line-*.pgm`
without `~` and the ten `mut-float-*` recipes: the float code, the recognizer and the forward pass
on edited weights) with the Go oracle on amd64 and with the oracle example built for aarch64 and run
natively on an arm64 runner (`ubuntu-24.04-arm`, with `-threads 1` and with the default threads), and
`diff -r` of the two trees must be empty. Go's `math.Exp` differs on arm64, so the Go side must run
on amd64: the CI job `go-arm64-data` runs it on an x86_64 runner and hands the inputs and Go outputs
to `parity-arm64` as an artifact. The Go run of the
subset is its own stage because the recognizer is shared by the `.pgm` inputs in name order and
carries its random generator from one to the next, so the subset must run on both sides to be
comparable. This is where `f64::mul_add` (a hardware FMA on aarch64, a libm call on x86_64) meets
Go's amd64 `math.Exp`.

Platform scope: the model is run on amd64 with FMA (the `ubuntu-24.04` runners have AVX and FMA), where Go's
`math.Exp` takes its FMA assembly path, and `nn::softmax_in_place` ports that path. Go's `math.Exp`
differs in the last bit between amd64 without FMA, arm64 (`exp_arm64.s`, another polynomial) and
the portable code, so byte-identity of the softmax and of the certainties holds against Go on amd64
with FMA only; against Go on another platform the tolerance is one ulp of those values. Nothing
else the crate computes depends on it (the imaging operators and the LSTM are fused-multiply-free
on both sides). The `arm64` job runs the Rust tests natively on an arm64 runner without the model and without Go;
Rust on aarch64 is compared with Go on amd64 by `parity-arm64`, on the subset above.

The model is tessdata_best 4.1.0 `eng.traineddata`, fetched with a SHA-256 check (it is not in the
repository, as in Go); the `model` stage fails the build without it, so the parity job never
skips the recognizer. `oracle -bench` and `examples/bench.rs` time the same model and corpus
(`bench.yml`, `tests/parity/bench.sh`).

## Inputs and reports

The input kind is the file extension; every report is text, written to `<name>.out`, or an error
class to `<name>.err`.

| Input | What it is | Report |
|---|---|---|
| `*.pgm` | a gray image: `line-<set>-<id>` (the 30 line images of `tests/data/lines`, 15 per arm, 8 of them also noised, `~n`, and 4 with the right half cut off, `~c`), `golden-gray` (the Leptonica scan, whole, noised and cropped), `rnd-<nnnn>` (400 generated: 1 to 80 pixels each way, noise, constant, gradients, dark blobs, text-like bars) | `dims`, `gray8` (the hash of `Bitmap::from_gray8` in Rust, `imaging.FromImage(*image.Gray)` in Go), then Otsu (threshold and bitmap), Sauvola at three windows, skew angle, rotate (by the skew, 0.05, on a gray bitmap, NaN, +Inf and 1e39, and by 0.9, 2, -2.5, 3.5, 5, pi/2, 1e6 and 1e9, which reach every octant of Go's `sin`/`cos` and its Payne-Hanek reduction), `scale_gray` at 0.5 1 1.7 2.5 0 -1, NaN, +Inf and 1e39, and on a bilevel one, dilate/erode/open/close, dilate on gray and with a 0x1 brick, `conn_comp` 4 8 6 and on gray, distance function, `seed_fill` 4 8 6 and with mismatched sizes, `raster_op` for the seven ops on bilevel and on gray (with clipped rectangles) and with mismatched depths, the grid (inserts of the components, five queries) and a grid with cell size 0. Each is a hash of the bitmap's pixels (or the float's bits) or `ERR <class>`. `line-*` and `rnd-*` also run the recognizer (so the 400 random bitmaps, 1 to 80 pixels each way, cover the upscaling factor above 1, the random fill of the rows `ScaleGray` leaves short, and the error for a scaled line below the network's minimum): the network output (hash, shape, scale, black, contrast), the greedy decode, the beam decode with and without the lexicon (every symbol with the bits of its certainty and rating) and `Recognize` (words, boxes, confidences). One recognizer is shared by all inputs, in name order, as in the Rust oracle, so the carried random generator is compared too. |
| `*.rgba` | an `image.RGBA`: `[w u32 LE][h u32 LE][w*h*4 bytes]`. `rgba-<nnnn>` (24: 1 to 64 pixels each way; 20 with random premultiplied pixels, colour channels never above the alpha, alpha anywhere in 0 to 255, a quarter opaque, and 4 with colour channels above the alpha) and `rgba-line-<n>` (the first three h36 lines, tinted and translucent) | `dims`, `rgba` (the hash of the gray bitmap: Go `imaging.FromImage(*image.RGBA)`, Rust `Bitmap::from_rgba8`), then the imaging steps of a `.pgm` on that bitmap and the recognizer (Go runs it on the `*image.RGBA` itself, so `FromImage` inside `Forward` and `Recognize` is compared too) |
| `*.nn` | 150 vectors of 1 to 64 f64 (scales 1 to 1e300, table edges, signed zero, subnormals) | the tabulated `tanh` and `logistic` and `softmax` in place (hash of the bits) and the first 8 values one at a time |
| `*.mut` | a recipe of edits (`trunc N`, `flip OFFSET BIT`, `set OFFSET VALUE`, `setf OFFSET HEXBITS` for the bits of an f64, `data HEX` to replace the bytes) to the model, or to one of its components (`part uni|rec|dawgp|dawgw|dawgn|lstm [swap]`, parsed alone, with the byte-swapped reader when `swap`) | whole model: `LoadModel` (version, the recognizer fields, sizes, the layer tree with weight statistics) and `NewRecognizer`, then the recognizer's report on a fixed noise line. Component: a hash of everything the parser keeps, or the class. A dawg component hashes every edge (id, next node, word-end flag), `EdgeChar` for every 7th node and eight ids, and `Contains`/`HasPrefix` on a few synthetic id lists and on 22 real words (and each without its last letter), mapped to ids by the eng unicharset |

Whole-model recipes: 15 cuts (header, offset table, the middle, the last bytes), 40 bit flips and 20
byte sets, mostly in the first 200 bytes (the offset table), 30 flips anywhere (almost all in the
weights, so most load and the forward pass runs on them) and 20 byte sets in the first 300 bytes
of the lstm component. Then 10 `mut-float-*` recipes (`setf`: NaN and infinite weights, found by value in the model's bytes, in the Tanh layer, in the first LSTM's CI (tanh) and GI (logistic) gates and in the softmax layer), 3 `mut-nest-N` (the lstm component replaced by a network of N nested Series layers, N = 64, 65, 500) and 2 `mut-*uni-notutf8` (a byte 0xFF in the unicharset, alone and in the whole model). For each of the six components: 13 cuts, 40 flips, 15 byte sets, 10 flips
under the byte-swapped reader, and the unedited component with each reader. The edits use an
xorshift64 with a fixed seed, so the inputs do not depend on the Go version.

## Error classes

Go's errors are `fmt.Errorf` strings that start with the package (`tessdata:`, `nn:`, `imaging:`,
`recog:`), and Go panics on bad arguments with a message of the same shape. The class is that
prefix. The Rust `Error` variants map one to one:

| Class | Go | Rust |
|---|---|---|
| `tessdata` | error or panic starting `tessdata:` | `Error::Tessdata` |
| `nn` | `nn:` | `Error::Network` |
| `imaging` | `imaging:` (panics on a bad depth, size, factor, op) | `Error::Imaging` |
| `recog` | `recog:` | `Error::Recog` |
| `pixel_limit` | the oracle refuses a plane over 1<<24 pixels | `Error::PixelLimit` (the Rust oracle sets `max_pixels` to 1<<24) |
| `model_limit` | the oracle refuses a model over 64 MiB | `Error::ModelLimit` |
| `alloc_limit` | no limit in Go: the oracle maps a child process that died of memory, or a recoder code range over 1<<30, onto it (`out_of_memory`, `recoder_code_range` in the rewrite table) | `Error::AllocLimit` |
| (none) | Go has no cancel and the crate's allocator does not refuse | `OutOfMemory`, `Cancelled`: either in the oracle output is a difference |

The oracle bounds its own memory: inputs over 1<<24 pixels or 64 MiB are refused before they are
decoded, the Go heap has a soft limit of 1 GiB, and sizes read from the files are checked against
the bytes that remain by the library itself.

There are no partial outputs to compare: no cadmus function returns a result together with an
error (the Go packages return a value or an error, and panic on bad arguments).

The run has 1287 inputs: 42 line images, 3 scan images, 400 random bitmaps, 27 RGBA images, 150
vectors, 163 whole-model recipes (10 of them the `mut-float-*` ones, 1 `mut-model-uni-notutf8` and
27 `mut-str-model-*`) and 502 component recipes (3 `mut-nest-N`, 1 `mut-uni-notutf8` and 18
`mut-str-part-*` among them). No whole-model recipe nests layers more than 64 deep: it would need a
container writer to splice a new `lstm` component in, so `layer_nesting` is mapped (and pinned by
`mut-nest-N`) for the `lstm` component only.

## Differences from Go that the oracle encodes

On every input of the run the Rust crate equals Go, except where Go has no defined behaviour on a
hostile model and the crate has limits or validation. The oracle maps Go's side onto the crate's:

- Go has no memory limit. An edited model can declare a tensor of terabytes, and Go's allocation
  failure is a fatal error, not a panic. Each `.mut` recipe therefore runs in a child process
  (`-one`) under a 3 GiB address-space limit that streams its report; a child that dies of memory
  is reported as `<step>: ERR alloc_limit` for the step it was in (`alloc_limit` for a component),
  which is what the crate's `max_alloc` (1 GiB) gives. `ParseRecoder` also allocates one byte per
  code value, so a code range over 1 GiB is reported as `alloc_limit` too.
- A NaN that reaches Go's tabulated tanh or logistic (a NaN weight, or `+Inf` and `-Inf` weights
  that sum to NaN) indexes the table with `int(NaN)`, a negative index, and panics in the forward
  pass (`runtime error: index out of range [-9223372036854775808]`). The crate builds the same
  network and fails the forward pass at the same point with an error (`Error::Network`, which the
  recognizer reports as `Error::Recog`); the oracle reports Go's panic as `forward: ERR recog`.
  A NaN that reaches only the softmax (or an activation with no table) gives output in both, with
  the same bits, and needs no rewrite; neither does an infinite weight that makes no NaN.
- `ScaleGray` checks only `factor <= 0`. For NaN, `+Inf` and a factor that overflows float32
  (`1e39`) Go computes a size of `MinInt64` and returns a bitmap with a negative width and height
  and no error. The crate returns `Error::Imaging` for them; the oracle reports Go's bitmap as
  `ERR imaging` (`scale_NaN`, `scale_Inf`, `scale_1e39`).
- Go has no limit on how deep layers nest (only 10000 children per stack); the crate stops at 64
  levels (`Error::Tessdata`). The oracle reports a Go network deeper than 64 as `tessdata`.
- A layer with a negative size (an edited `half_x`) builds in Go and panics with a runtime error
  in the forward pass; the crate rejects it when it builds the network. A Go forward pass that ends
  in a runtime error (class `other`) after a successful `NewRecognizer` is reported as
  `recognizer: ERR recog`. Only a Go runtime panic (`FWDERR runtime error:` on the child's stderr)
  is rewritten; any other forward error stays as Go reported it.
- The crate requires a UTF-8 unicharset, Go accepts any bytes: the oracle reports `tessdata`
  (`recog` from `NewRecognizer`) for an edited model whose unicharset is not UTF-8.
- Layer names, the spec string and the version string are decoded lossily by the crate
  (`String::from_utf8_lossy`: one U+FFFD per maximal ill-formed subpart, so FF FE gives two and
  E2 82 followed by a valid byte gives one). The oracle does the same to the Go strings before it
  prints them (`lossy` in `report.go`, which is neither `ToValidUTF8` nor a rune loop) and counts
  each string it changes as `lossy_string`; the layer names are decoded before Go's `Tree` quotes
  them with `%q`, and the crate quotes with the same rules (`tessdata::quote`, Go's
  `strconv.Quote`: `\x01`, `\x7f`, `\u00a0`, `\u200b`, a combining mark literal).
- Go's tabulated `tanh` and `logistic` panic on a NaN (`int(NaN)` indexes the table with a negative
  number); the crate's `nn::tanh` and `nn::logistic` return NaN. The `.nn` vectors include NaN,
  `+Inf` and `-Inf`; a vector with a NaN is computed element by element and a call that panics
  gives NaN, counted as `table_nan_panic`. `+Inf` and `-Inf` need no rewrite.
- NaN payloads are not compared (every NaN hashes the same).

The oracle counts how often each rewrite fires and prints the totals on stderr after the run
(`rewrites: forward_nan_in_table=N forward_runtime_error=N layer_nesting=N lossy_string=N
out_of_memory=N recoder_code_range=N scale_negative_size=N table_nan_panic=N
unicharset_not_utf8=N`), followed by one `rewrite <kind>: <inputs>` line each with the first six
inputs that fired it, so the go-run stage log shows them. Every difference above has at least one
input that fires it (counts of the last run: `forward_nan_in_table=3 forward_runtime_error=1
layer_nesting=2 lossy_string=13 out_of_memory=1 recoder_code_range=1 scale_negative_size=1416
table_nan_panic=40 unicharset_not_utf8=16`):

| Rewrite | Inputs that pin it |
|---|---|
| `forward_nan_in_table` | `mut-float-tanh-nan`, `mut-float-tanh-infpair`, `mut-float-lstm-ci-nan` (the same family has the opposite direction too: `mut-float-tanh-biasninf`, `mut-float-lstm-gi-*`, `mut-float-softmax-*` give output in both) |
| `scale_negative_size` | `scale_NaN`, `scale_Inf`, `scale_1e39` on every image input (472 inputs, three steps each) |
| `layer_nesting` | `mut-nest-65`, `mut-nest-500` (and `mut-nest-64`, which loads in both) |
| `lossy_string` | `mut-model-0108`, `mut-model-0119`, `mut-model-0120` (the same kind of byte set, here in a layer name or the spec string) and the `mut-str-*` recipes: bytes `FF FE` and `E2 82` in a layer name (`mut-str-part-name-*`, `mut-str-model-name-*`), in the spec string (`mut-str-part-spec-*`, `mut-str-model-spec-*`) and in the version (`mut-str-model-version-*`); the control, DEL, no-break space, zero-width space and combining-mark variants of the same recipes (`ctl01`, `del7f`, `nbsp`, `zwsp`, `comb`) and the default-ignorable ones that Go prints literally (`vs16` is U+FE0F, `hf` the Hangul filler U+3164) pin the quoting and fire no rewrite |
| `table_nan_panic` | the `vec-*.nn` vectors that hold a NaN |
| `forward_runtime_error` | `mut-model-0110` (one of the 20 random byte sets in the first 300 bytes of the lstm component, here an edited layer size: Go loads it and its forward pass dies of a runtime panic; the rule fires on any `FWDERR runtime error:`, and this is the only input of the run that does) |
| `out_of_memory` | `mut-model-0123` (the same kind of edit, here a size that makes the child process run out of its 3 GiB address space) |
| `recoder_code_range` | `mut-rec-0247` (a random edit of the recoder component that gives a code range over 1<<30) |
| `unicharset_not_utf8` | `mut-uni-notutf8` (the component), `mut-model-uni-notutf8` (the whole model: `load` and `recognizer`), and the random recipes that happen to break UTF-8 |

## Go tests and generators

The Go repo has 189 test functions ([testing.md](testing.md) has the per-package counts) and none of
them uses a random generator: the inputs are hand-built bitmaps and networks, the Leptonica
goldens and the line corpus. The unit tests are ported as `src/*/tests.rs`. The inputs above are
therefore new generators, not ports: the random bitmaps and vectors, the line and scan mutations
and the model mutations. Go provides the expected bytes for all of them.

Not generated: the hand-built networks of `internal/nn` and `internal/recog` tests (the Rust
tests have them, with the Go expected values), and the corpus comparison with Tesseract itself
(`.psm13.txt`), which is checked in `src/recog/tests.rs`.
