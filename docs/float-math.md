# Float math

Rust does not fuse `a * b + c`, so the scalar f64 code matches Go on amd64 (where Go does not
fuse either) without the `float64(x*y)` wrapping Go needs on arm64. Go's `math.Exp` (softmax) is
the assembly version on amd64, which differs from the portable one and from libm in the last bit;
`nn::softmax_in_place` ports the amd64 FMA path with `f64::mul_add`. Go's `math.Sin`, `Cos`,
`Tan` (deskew) are pure Go and are ported as such (`imaging/gomath.rs`).

Byte-identity of the softmax and of the certainty (`math.Exp`) is therefore with Go on amd64 with
FMA (AVX and FMA present), which is where the `parity` job runs. Go's own `math.Exp` differs in the
last bit between amd64 without FMA, arm64 and the portable code, so on any other Go platform the
two can differ by one ulp there; [parity.md](parity.md) states the scope. The `arm64` job runs the Rust
tests natively on an arm64 runner and has no Go to compare with; the `parity-arm64` job builds the oracle example for
aarch64 and compares its output on that runner with the Go output (from amd64) of the same inputs (a subset of them),
which is where the hardware FMA of `f64::mul_add` meets Go's amd64 result.
