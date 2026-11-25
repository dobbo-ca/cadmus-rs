// Command oracle is the Go side of the cadmus-rs parity job. It runs the cadmus packages over
// every file of an input directory and writes, per input, `<name>.out` (a text report) or
// `<name>.err` (an error class: a short stable token). examples/oracle.rs does the same with the
// Rust crate; the two output trees must be equal.
//
//	oracle -model eng.traineddata -gen tests/data /inputs     write the inputs
//	oracle -model eng.traineddata /inputs /go-out           run them
//	oracle -bench -model eng.traineddata tests/data           time it (tests/parity/bench.sh)
//
// The input kind is the file extension: .pgm (an image: the imaging operators, and for names
// starting `line-` or `rnd-` the recognizer), .rgba (an image.RGBA: the same, with the
// recognizer), .nn (f64 vectors), .mut (a recipe: edits applied to the model, or to one of its
// components, which is then parsed alone).
//
// Memory is bounded: images over 1<<24 pixels are refused as `pixel_limit` before they are
// decoded, a model over 64 MiB as `model_limit` (the Rust oracle runs with the same limits), and
// the Go heap has a soft limit.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
)

const (
	maxPixels     = 1 << 24
	maxModelBytes = 64 << 20
)

var modelPath = flag.String("model", "", "tessdata_best eng.traineddata")
var gen = flag.Bool("gen", false, "write the inputs instead of running them")
var one = flag.String("one", "", "child mode: run one .mut input under a memory rlimit and print `O:report` or `E:class`")
var topng = flag.Bool("topng", false, "convert a PGM to a gray PNG (for cadmusdump -activations): oracle -topng in.pgm out.png")
var benchmark = flag.Bool("bench", false, "time the recognizer and the imaging operators: oracle -bench -model m <testdata>")

func main() {
	flag.Parse()
	debug.SetMemoryLimit(1 << 30)
	if *topng {
		pgmToPNG(flag.Arg(0), flag.Arg(1))
		return
	}
	if *benchmark {
		model, err := os.ReadFile(*modelPath)
		if err != nil {
			fatal(err)
		}
		bench(flag.Arg(0), model)
		return
	}
	if *one != "" {
		runOne(*one, model0())
		return
	}
	if flag.NArg() < 2 {
		fmt.Fprintln(os.Stderr, "usage: oracle [-model m] [-gen] <in> <out>")
		os.Exit(2)
	}
	in, out := flag.Arg(0), flag.Arg(1)
	model := model0()
	if err := os.MkdirAll(out, 0o755); err != nil {
		fatal(err)
	}
	if *gen {
		generate(in, out, model)
		return
	}
	names, err := os.ReadDir(in)
	if err != nil {
		fatal(err)
	}
	var files []string
	for _, e := range names {
		if !e.IsDir() {
			files = append(files, e.Name())
		}
	}
	sort.Strings(files)
	o := newOracle(model)
	o.dir = in
	n := 0
	for _, f := range files {
		data, err := os.ReadFile(filepath.Join(in, f))
		if err != nil {
			fatal(err)
		}
		currentInput = f
		report, class := o.run(f, data)
		if class != "" {
			write(out, f+".err", class+"\n")
		} else {
			write(out, f+".out", report)
		}
		if n++; n%2000 == 0 {
			fmt.Fprintf(os.Stderr, "%d/%d\n", n, len(files))
		}
	}
	fmt.Fprintf(os.Stderr, "%d inputs\n", n)
	printRewrites()
}

// inChild is set in the child of mutantChild.
var inChild bool

// rewrites counts the inputs on which the oracle replaced Go's outcome with the Rust crate's
// (docs/parity.md, "Differences from Go that the oracle encodes"), by kind. A child reports
// each on stderr and the parent adds them up.
var rewrites = map[string]int{}

// currentInput is the file being run, and rewriteInputs the first few inputs of each kind, which
// the log prints so that docs/parity.md can name the inputs that pin each rewrite.
var currentInput string
var rewriteInputs = map[string][]string{}

func noteRewrite(kind string) {
	if inChild {
		fmt.Fprintln(os.Stderr, "REWRITE", kind)
		return
	}
	addRewrite(kind)
}

func addRewrite(kind string) {
	rewrites[kind]++
	if in := rewriteInputs[kind]; len(in) < 6 && (len(in) == 0 || in[len(in)-1] != currentInput) {
		rewriteInputs[kind] = append(in, currentInput)
	}
}

func printRewrites() {
	var kinds []string
	for k := range rewrites {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	fmt.Fprint(os.Stderr, "rewrites:")
	for _, k := range kinds {
		fmt.Fprintf(os.Stderr, " %s=%d", k, rewrites[k])
	}
	fmt.Fprintln(os.Stderr)
	for _, k := range kinds {
		fmt.Fprintf(os.Stderr, "rewrite %s: %s\n", k, strings.Join(rewriteInputs[k], " "))
	}
}

func pgmToPNG(in, out string) {
	data, err := os.ReadFile(in)
	if err != nil {
		fatal(err)
	}
	b, err := parsePGM(data)
	if err != nil {
		fatal(err)
	}
	g := image.NewGray(image.Rect(0, 0, b.Width, b.Height))
	for y := 0; y < b.Height; y++ {
		for x := 0; x < b.Width; x++ {
			g.SetGray(x, y, color.Gray{Y: b.At(x, y)})
		}
	}
	f, err := os.Create(out)
	if err != nil {
		fatal(err)
	}
	if err := png.Encode(f, g); err != nil {
		fatal(err)
	}
	if err := f.Close(); err != nil {
		fatal(err)
	}
}

func model0() []byte {
	if *modelPath == "" {
		return nil
	}
	b, err := os.ReadFile(*modelPath)
	if err != nil {
		fatal(err)
	}
	return b
}

func write(dir, name, s string) {
	if err := os.WriteFile(filepath.Join(dir, name), []byte(s), 0o644); err != nil {
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "oracle:", err)
	os.Exit(1)
}

// run dispatches on the extension. It returns the report, or a top-level error class.
func (o *oracle) run(name string, data []byte) (report, class string) {
	ext := name[strings.LastIndexByte(name, '.')+1:]
	switch ext {
	case "pgm":
		return o.image(name, data)
	case "rgba":
		return o.rgba(data)
	case "nn":
		return o.vectors(data)
	case "mut":
		return o.mutantChild(name)
	}
	return "", "unknown_input"
}
