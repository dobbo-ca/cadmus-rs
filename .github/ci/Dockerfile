# CI image: every check runs in a BuildKit build of this file, so CI and a local `docker buildx build` use the same toolchain.
# The one dependency (rayon) is fetched in the `deps` stage, before the source is copied, into the
# registry cache mount that the stages that build use (an id of its own, like the target mounts), so a
# commit that does not touch a manifest or a lockfile does not fetch again. The fuzz crate
# (libfuzzer-sys, and rayon through the library) has its own `cargo fetch` layer in `fuzz-tools`.
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu
WORKDIR /src

# The manifest and lockfile of the crate, with a stub target, so the fetch only runs again when
# they change. The fuzz and tool stages do not build on it.
FROM toolchain AS deps
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry \
    mkdir -p src && touch src/lib.rs && cargo fetch --locked

FROM deps AS src
COPY . .

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The registry and target dirs are cache mounts, so they persist across builds.
# Every target mount has an id of its own (repo and stage): BuildKit keys a mount without an id by
# its path, so every build on a shared BuildKit daemon would share /src/target.
FROM src AS ci
# The aarch64 release build is of the library only: linking cadmusdump would need the cross linker.
# The arm64 stage runs on a native arm64 runner and builds everything for its own architecture.
# The tessdata_best eng model (15 MB, not in the repository). The tests that need it skip without
# it, so a failed fetch only loses them; with it they check the loader, the network and both
# decoders on the real model and the line corpus.
RUN curl -fsSL --retry 3 -o tests/data/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/eng.traineddata \
    && echo "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba  tests/data/eng.traineddata" | sha256sum -c - \
    || { echo "model fixture not fetched: the model tests will skip"; rm -f tests/data/eng.traineddata; }
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-ci,target=/src/target \
    cargo fmt --check
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-ci,target=/src/target \
    cargo clippy --locked --all-targets -- -D warnings
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-ci,target=/src/target \
    cargo test --locked -- --nocapture
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-ci,target=/src/target \
    cargo build --locked --release --target aarch64-unknown-linux-gnu --lib

# The tessdata_best eng model, for the stages that must not skip without it (parity, bench, fuzz):
# a failed fetch or a wrong hash fails the build.
FROM rust:1.99.1-bookworm AS model
RUN curl -fsSL --retry 3 -o /eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/eng.traineddata \
    && echo "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba  /eng.traineddata" | sha256sum -c -

# aarch64 tests, built and run natively (the arm64 job runs on an arm64 runner). The model is not
# fetched, so the tests that need it skip; the arithmetic they pin is compared with Go by the
# parity job. There is no SIMD tier, so there is one run, of only the tests that
# the target changes: the float code, where aarch64 fuses multiply-adds and rounds differently
# (nn:: holds the activation functions and the matrix kernels, imaging:: the scaling and
# thresholding, recog:: the normalization, beam search and the forward pass on a synthetic network).
# The rest (the container parser, the dictionaries) is plain integer and logic code that the full
# suite runs natively in `ci`; Go parity on the whole is the parity stage.
FROM src AS arm64
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-arm64,target=/src/target \
    cargo test --locked --lib -- nn:: imaging:: recog:: tessdata::network

# Go parity: the Go oracle writes the inputs and runs them; the Rust example runs the same inputs;
# the two output trees must be equal.
FROM golang:1.27-bookworm AS go-oracle
ENV GOTOOLCHAIN=local CGO_ENABLED=0
WORKDIR /src
COPY tests/parity/go tests/parity/go
# oracle-v4: the same oracle built with GOAMD64=v4, for the v4 column of the bench (docs/performance.md).
RUN --mount=type=cache,id=cadmus-rs-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && go build -o /out/oracle . \
    && GOAMD64=v4 go build -o /out/oracle-v4 . \
    && cd ../upstream && go build -o /out/cadmusdump ./cmd/cadmusdump

FROM go-oracle AS go-run
COPY tests/data tests/data
COPY --from=model /eng.traineddata /model/eng.traineddata
RUN /out/oracle -model /model/eng.traineddata -gen tests/data /inputs \
    && ls /inputs | wc -l && du -sh /inputs
RUN /out/oracle -model /model/eng.traineddata /inputs /go-out && ls /go-out | wc -l && du -sh /go-out
# cadmusdump with each flag (and with all three), and its activation dump of one line image (the Go
# tool reads a PNG, the Rust one a PGM: the oracle writes the PNG from the PGM).
RUN mkdir /go-dump && cd /go-dump \
    && /out/cadmusdump /model/eng.traineddata > default.txt \
    && /out/cadmusdump -unicharset /model/eng.traineddata > unicharset.txt \
    && /out/cadmusdump -recoder /model/eng.traineddata > recoder.txt \
    && /out/cadmusdump -dawgs /model/eng.traineddata > dawgs.txt \
    && /out/cadmusdump -unicharset -recoder -dawgs /model/eng.traineddata > all.txt \
    && /out/oracle -topng /src/tests/data/lines/h36/0001.pgm /tmp/line.png \
    && /out/cadmusdump -activations /model/eng.traineddata /tmp/line.png > activations.txt \
    && wc -c *.txt

# The Go trees are bind mounts from go-run, not COPY: a COPY of the output tree after `COPY . .`
# would run again on every commit. The mounts are `rw` (the writes are thrown away after the RUN):
# BuildKit keys a read-only mount by a hash of its content, a rw mount by go-run's own cache key.
# The Rust output is deleted in the same RUN, so the layer (and the cache export) stays empty.
FROM src AS parity
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=model /eng.traineddata /model/eng.traineddata
# The Rust oracle runs four times on the same inputs: with one thread (`-threads 1`, no pool, nothing
# spawned), with the default `max_threads` (every CPU) on the crate's shared pool (the default `-pool
# shared`: no caller pool), on a caller's pool (`-pool caller`) and on workers the call builds for
# itself (`-pool percall`), and every output must equal the Go output.
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-parity,target=/src/target \
    --mount=type=bind,from=go-run,source=/inputs,target=/inputs,rw \
    --mount=type=bind,from=go-run,source=/go-out,target=/go-out,rw \
    cargo build --locked --release --example oracle \
    && echo "$(ls /inputs | wc -l) inputs, $(ls /go-out | wc -l) outputs" \
    && [ "$(ls /go-out | wc -l)" -ge 1000 ] \
    && for t in "-threads 1" "-threads 0" "-threads 0 -pool caller" "-threads 0 -pool percall"; do \
        echo "oracle $t" \
        && target/release/examples/oracle $t -model /model/eng.traineddata /inputs /rust-out \
        && [ "$(ls /go-out | wc -l)" -eq "$(ls /rust-out | wc -l)" ] \
        && { diff -rq /go-out /rust-out >/tmp/diff.txt || { head -60 /tmp/diff.txt; wc -l /tmp/diff.txt; diff -r /go-out /rust-out | head -120; sh tests/parity/explain.sh; exit 1; }; } \
        && rm -rf /rust-out || exit 1; \
    done
# cadmusdump: the same flags on the same model path, byte for byte.
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-parity,target=/src/target \
    --mount=type=bind,from=go-run,source=/go-dump,target=/go-dump,rw \
    cargo build --locked --release --bin cadmusdump \
    && D=/src/target/release/cadmusdump && M=/model/eng.traineddata && mkdir /rust-dump \
    && $D $M > /rust-dump/default.txt \
    && $D -unicharset $M > /rust-dump/unicharset.txt \
    && $D -recoder $M > /rust-dump/recoder.txt \
    && $D -dawgs $M > /rust-dump/dawgs.txt \
    && $D -unicharset -recoder -dawgs $M > /rust-dump/all.txt \
    && $D -activations $M tests/data/lines/h36/0001.pgm > /rust-dump/activations.txt \
    && { diff -r /go-dump /rust-dump | head -60 | cut -c1-200; diff -rq /go-dump /rust-dump; } \
    && rm -rf /rust-dump

# Go parity on aarch64: the Go oracle runs a subset of the inputs (the `.nn` vectors, the gray-scan,
# ten random bitmaps and two lines with the recognizer, and the `mut-float-*` recipes, which build
# the network and run a forward pass), and the oracle example built natively for aarch64 runs the
# same subset, with one thread and with the default threads (the images over 131,072 pixels
# split into bands only with more than one). The two output trees must be equal: the float
# code (Go's amd64 `math.Exp` path ported with `f64::mul_add`, which is a hardware FMA here and a
# libm call on x86_64) is compared with Go's output on amd64. So the Go run happens on an x86_64
# runner (the stages go-run-arm64 and go-arm64-data; the job go-arm64-data of ci.yml writes the
# subset and its Go outputs to ./go-arm64-data), and the parity-arm64 job on an arm64 runner
# downloads them into the build context. The Go run is its own stage because
# the recognizer carries its random generator from one input to the next, so a subset has to be
# run on both sides to be comparable.
FROM go-oracle AS go-run-arm64
COPY --from=model /eng.traineddata /model/eng.traineddata
RUN --mount=type=bind,from=go-run,source=/inputs,target=/inputs,rw \
    mkdir /inputs-arm64 && cd /inputs \
    && cp vec-*.nn mut-float-*.mut golden-gray.pgm rnd-000?.pgm /inputs-arm64/ \
    && cp $(ls line-*.pgm | grep -v '~' | head -2) /inputs-arm64/ \
    && ls /inputs-arm64 | wc -l \
    && /out/oracle -model /model/eng.traineddata /inputs-arm64 /go-out-arm64 && ls /go-out-arm64 | wc -l

FROM scratch AS go-arm64-data
COPY --from=go-run-arm64 /inputs-arm64 /inputs-arm64
COPY --from=go-run-arm64 /go-out-arm64 /go-out-arm64

# Needs ./go-arm64-data (the output of the go-arm64-data stage), which `COPY . .` brings in.
FROM src AS parity-arm64
COPY --from=model /eng.traineddata /model/eng.traineddata
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-parity-arm64,target=/src/target \
    inp=/src/go-arm64-data/inputs-arm64 go=/src/go-arm64-data/go-out-arm64 \
    && cargo build --locked --release --example oracle \
    && echo "$(ls $inp | wc -l) inputs, $(ls $go | wc -l) outputs" \
    && [ "$(ls $go | wc -l)" -ge 150 ] \
    && for t in 1 0; do \
        echo "oracle -threads $t" \
        && target/release/examples/oracle -threads $t -model /model/eng.traineddata $inp /rust-out-arm64 \
        && [ "$(ls $go | wc -l)" -eq "$(ls /rust-out-arm64 | wc -l)" ] \
        && { diff -rq $go /rust-out-arm64 >/tmp/diff.txt || { head -60 /tmp/diff.txt; diff -r $go /rust-out-arm64 | head -120; exit 1; }; } \
        && rm -rf /rust-out-arm64 || exit 1; \
    done

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The workflow
# runs it with `docker run`, so a BuildKit cache hit cannot skip the fuzzing. The model the
# recognizer targets read is in the image. The fuzz-arm64 job builds this same stage on a native
# arm64 runner.
FROM toolchain AS fuzz-tools
COPY --from=model /eng.traineddata /src/tests/data/eng.traineddata
ENV CADMUS_MODEL=/src/tests/data/eng.traineddata
RUN rustup toolchain install nightly --profile minimal \
    && cargo install cargo-fuzz --locked --version 0.13.1
# The fuzz crate's dependencies are fetched before the source is copied, with stub targets, so a
# source change does not download them again.
COPY Cargo.toml Cargo.lock /src/
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock /src/tests/fuzz/
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in $(sed -n 's|^path = "fuzz_targets/\(.*\)\.rs"|\1|p' tests/fuzz/Cargo.toml); do touch tests/fuzz/fuzz_targets/$t.rs; done \
    && cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml

FROM fuzz-tools AS fuzz
COPY . .
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz

# Benchmark (bench.yml): the Go oracle (-bench) and the Rust bench example in one image, each also
# built for x86-64-v4 (oracle-v4: GOAMD64=v4; bench-v4: the RUSTFLAGS of docs/performance.md, in a
# target dir of its own). The sources are touched first, so a target cache that holds a newer build
# of other sources cannot be taken as up to date.
FROM src AS bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=go-oracle /out/oracle-v4 /usr/local/bin/oracle-v4
COPY --from=model /eng.traineddata /src/tests/data/eng.traineddata
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-bench,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-bench-v4,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --locked --release --example bench \
    && cp target/release/examples/bench /usr/local/bin/bench-v4

# PGO and BOLT (pgo.yml, dispatch only; docs/performance.md): cargo-pgo builds the bench example
# instrumented, trains it on the bench inputs, rebuilds it with the profile, and, when BOLT is
# installed, once more through BOLT. The tools are installed before `COPY . .`, so a new commit
# does not install them again. BOLT comes from apt.llvm.org (bookworm has none of its own); if the
# repository or the package is missing the image still builds and the job reports PGO only.
FROM toolchain AS pgo-tools
RUN (rustup component add llvm-tools || rustup component add llvm-tools-preview) \
    && cargo install cargo-pgo --locked
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && { wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
         && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main" > /etc/apt/sources.list.d/llvm.list \
         && apt-get update && apt-get install -y --no-install-recommends bolt-19 libbolt-19-dev \
         && ln -s /usr/lib/llvm-19/bin/llvm-bolt /usr/local/bin/llvm-bolt \
         && ln -s /usr/lib/llvm-19/bin/merge-fdata /usr/local/bin/merge-fdata \
         && ln -s "$(find /usr/lib/llvm-19 -name libbolt_rt_instr.a | head -n 1)" /usr/local/lib/libbolt_rt_instr.a \
         && test -s /usr/local/lib/libbolt_rt_instr.a; } \
    || echo "BOLT is not available: the PGO job reports PGO only"; \
    rm -rf /var/lib/apt/lists/*

# bench (default build), bench-pgo and, when BOLT worked, bench-bolt (PGO then BOLT), all in
# /usr/local/bin with the Go oracle; tests/parity/pgo.sh times them. The profiles of an earlier run
# stay in the cache mount, so they are deleted first. The training runs the bench with 1 thread and
# with the default threads, so both paths are in the PGO profile; the BOLT profile is trained on 1
# thread only.
FROM pgo-tools AS pgo
COPY . .
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=model /eng.traineddata /src/tests/data/eng.traineddata
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry --mount=type=cache,id=cadmus-rs-target-pgo,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && rm -rf target/pgo-profiles target/bolt-profiles && find target -name 'bench-bolt-*' -delete \
    && host=$(rustc -vV | sed -n 's/^host: //p') && bin=target/$host/release/examples/bench \
    && M=tests/data/eng.traineddata \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench \
    && cargo pgo build -- --locked --example bench \
    && cargo pgo run -- --locked --example bench -- $M tests/data 1 \
    && cargo pgo run -- --locked --example bench -- $M tests/data \
    && cargo pgo optimize build -- --locked --example bench && cp $bin /usr/local/bin/bench-pgo \
    && if command -v llvm-bolt >/dev/null && command -v merge-fdata >/dev/null; then \
         { cargo pgo bolt build --with-pgo -- --locked --example bench \
           && "$(dirname $bin)/bench-bolt-instrumented" $M tests/data 1 >/dev/null \
           && cargo pgo bolt optimize --with-pgo -- --locked --example bench \
           && cp "$(find target -type f -name 'bench-bolt-optimized' | head -n 1)" /usr/local/bin/bench-bolt; } \
         || echo "BOLT failed: the job reports PGO only"; \
       else echo "no BOLT: the job reports PGO only"; fi

# Real-file OCR gate (quality.yml; docs/quality.md). The label, macro and thumbnail pages of the
# OpenSlide public test slides are decoded with tiff-rs (and jpeg-rs) and cut into text-line crops
# once (tests/quality/labels); the Go oracle and examples/oracle.rs then run the same crops and their
# outputs (text, words, confidences, the activations' hash) must be byte-identical
# (tests/quality/gate.sh): with 1 thread, the default threads and, on a CPU with AVX-512, the
# x86-64-v4 build. No secrets: every input is public.
# The slides: https://openslide.cs.cmu.edu/download/openslide-testdata/, each checked against the
# SHA-256 in that directory's index.json.
FROM toolchain AS quality-data
RUN mkdir /data && cd /data && base=https://openslide.cs.cmu.edu/download/openslide-testdata \
    && for e in \
         ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7:Aperio/CMU-1-Small-Region.svs \
         00a3d54482cd707abf254fe69dccc8d06b8ff757a1663f1290c23418c480eb30:Aperio/CMU-1.svs \
         9a1923cd9bcb260ba4d99d64f8d6e32550648c332ba48817f920662f3a513420:Aperio/CMU-1-JP2K-33005.svs \
         6205ccf75a8fa6c32df7c5c04b7377398971a490fb6b320d50d91f7ba6a0e6fd:Aperio/JP2K-33003-1.svs \
         edf4a1ccf395c7000ae93ad3b44c07d97043810e00be0c1d167dd09bbe436e46:Hamamatsu/CMU-1.ndpi \
         63a3c00fef5215497a9725cf19092a93f7f5ff855ce8561af74b087631831a97:Leica/Leica-1.scn \
         a9037f7d683dc0a95e2e5f33bceee71f0449042e70e6d1421ca580bcd97a023a:Leica/Leica-Fluorescence-1.scn \
         73a7e89bc15576587c3d68e55d9bf92f09690280166240b48ff4b48230b13bcd:Generic-TIFF/CMU-1.tiff \
         6a743d3bbf721a63f1fc2920f41ddcca43571346665114c6827e63efe034a0b0:Philips-TIFF/Philips-4.tiff \
         2eb05cd8f628c36408c3fde09beeeebaaae90181c4f840d5475ec7594bdcbbb6:Ventana/Ventana-1.bif; do \
         sum=${e%%:*} p=${e#*:} && mkdir -p "$(dirname "$p")" \
         && curl -fsSL --retry 3 -o "$p" "$base/$p" && echo "$sum  $p" | sha256sum -c - || exit 1; \
       done

# The line cutter, built from its own manifest and lockfile only (tiff-rs, and through it jpeg-rs
# and deflate-rs, are public git dependencies).
FROM toolchain AS quality-labels
COPY tests/quality/labels /labels
RUN cd /labels && cargo build --locked --release \
    && cp target/release/cadmus-quality-labels /usr/local/bin/labels \
    && cd / && rm -rf /labels /usr/local/cargo/git /usr/local/cargo/registry

# One RUN: the slides are a bind mount of quality-data, and the crops and outputs are deleted at the
# end, so the layer holds only /report. The x86-64-v4 run is added only on a CPU with AVX-512.
FROM src AS quality
COPY --from=quality-labels /usr/local/bin/labels /usr/local/bin/labels
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=model /eng.traineddata /model/eng.traineddata
RUN --mount=type=cache,id=cadmus-rs-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=cadmus-rs-target-quality,sharing=locked,target=/src/target \
    --mount=type=bind,from=quality-data,source=/data,target=/data \
    find src examples tests -type f -exec touch {} + \
    && cargo build --locked --release --example oracle \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" CARGO_TARGET_DIR=target/v4 \
       cargo build --locked --release --example oracle \
    && if grep -qw avx512bw /proc/cpuinfo; then export RS_V4=target/v4/release/examples/oracle; fi \
    && sh tests/quality/gate.sh target/release/examples/oracle /model/eng.traineddata /report \
       /data/Aperio/CMU-1-Small-Region.svs /data/Aperio/CMU-1.svs /data/Aperio/CMU-1-JP2K-33005.svs \
       /data/Aperio/JP2K-33003-1.svs /data/Hamamatsu/CMU-1.ndpi /data/Leica/Leica-1.scn \
       /data/Leica/Leica-Fluorescence-1.scn /data/Generic-TIFF/CMU-1.tiff \
       /data/Philips-TIFF/Philips-4.tiff /data/Ventana/Ventana-1.bif \
    && cat /report/summary.md | head -40

FROM scratch AS quality-report
COPY --from=quality /report /
