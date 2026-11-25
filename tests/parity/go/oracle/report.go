package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/dobbo-ca/cadmus/internal/imaging"
	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/recog"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

var (
	errPixel = errors.New("pixel_limit")
	errModel = errors.New("model_limit")
)

// class maps an error to a short stable token: the package prefix of its message, which the Rust
// Error variants mirror (see docs/parity.md).
func class(err error) string {
	switch {
	case errors.Is(err, errPixel), errors.Is(err, errModel):
		return err.Error()
	}
	s := err.Error()
	if os.Getenv("ORACLE_MSG") != "" {
		fmt.Fprintln(os.Stderr, "[go]", s)
	}
	for _, p := range []string{"tessdata", "nn", "imaging", "recog"} {
		if strings.HasPrefix(s, p+":") {
			return p
		}
	}
	return "other"
}

type oracle struct {
	model []byte
	rec   *recog.Recognizer
	dir   string // the input dir, for mutantChild
}

func newOracle(model []byte) *oracle {
	o := &oracle{model: model}
	if model != nil {
		r, err := recog.NewRecognizer(model)
		if err != nil {
			fatal(fmt.Errorf("the model does not load: %w", err))
		}
		o.rec = r
	}
	return o
}

// fnv is FNV-1a 64.
type fnv struct{ h uint64 }

func newFnv() *fnv { return &fnv{14695981039346656037} }
func (f *fnv) u8(b byte) {
	f.h ^= uint64(b)
	f.h *= 1099511628211
}
func (f *fnv) u32(v uint32) {
	for i := 0; i < 4; i++ {
		f.u8(byte(v >> (8 * i)))
	}
}
func (f *fnv) u64(v uint64) {
	f.u32(uint32(v))
	f.u32(uint32(v >> 32))
}
func (f *fnv) str(s string) {
	f.u32(uint32(len(s)))
	for i := 0; i < len(s); i++ {
		f.u8(s[i])
	}
}

// f64 and f32 canonicalise NaN so its payload is not compared.
func f64(x float64) uint64 {
	if x != x {
		return 0x7ff8000000000000
	}
	return math.Float64bits(x)
}
func f32(x float32) uint32 {
	if x != x {
		return 0x7fc00000
	}
	return math.Float32bits(x)
}

func hashBitmap(b *imaging.Bitmap) string {
	f := newFnv()
	for y := 0; y < b.Height; y++ {
		for x := 0; x < b.Width; x++ {
			f.u8(b.At(x, y))
		}
	}
	return fmt.Sprintf("%dx%dx%d %016x", b.Width, b.Height, b.Depth, f.h)
}

// outbuf is a report under construction. In the child of mutantChild it is also streamed to
// stdout as it grows, so that if the child dies (Go's out-of-memory is fatal, not a panic) the
// parent still has everything up to that point.
type outbuf struct {
	sb   strings.Builder
	live io.Writer
}

func (o *outbuf) Write(p []byte) (int, error) {
	if o.live != nil {
		_, _ = o.live.Write(p)
	}
	return o.sb.Write(p)
}

func (o *outbuf) WriteString(s string) (int, error) { return o.Write([]byte(s)) }
func (o *outbuf) String() string                    { return o.sb.String() }

// childStep tells the parent which step the child is in, on stderr, when it is a child.
func childStep(name string) {
	if childLive != nil {
		fmt.Fprintln(os.Stderr, "STEP", name)
	}
}

var childLive io.Writer

func safely(f func() (string, error)) (s string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r)
		}
	}()
	return f()
}

// step runs f, catching Go's panics on bad arguments (Rust returns them as errors), and writes
// `name: result` or `name: ERR class`.
func step(b *outbuf, name string, f func() (string, error)) {
	childStep(name)
	s, err := safely(f)
	if err != nil {
		if inChild && name == "forward" {
			fmt.Fprintln(os.Stderr, "FWDERR", strings.ReplaceAll(err.Error(), "\n", " "))
		}
		fmt.Fprintf(b, "%s: ERR %s\n", name, class(err))
		return
	}
	fmt.Fprintf(b, "%s: %s\n", name, s)
}

func bm(f func() *imaging.Bitmap) func() (string, error) {
	return func() (string, error) { return hashBitmap(f()), nil }
}

// parsePGM reads the P5 files the oracle writes (and tests/data/lines holds).
func parsePGM(data []byte) (*imaging.Bitmap, error) {
	if len(data) < 2 || string(data[:2]) != "P5" {
		return nil, errors.New("imaging: not a P5 file")
	}
	pos := 2
	tok := func() (int, bool) {
		for pos < len(data) && (data[pos] == ' ' || data[pos] == '\n' || data[pos] == '\t' || data[pos] == '\r') {
			pos++
		}
		start := pos
		for pos < len(data) && data[pos] >= '0' && data[pos] <= '9' {
			pos++
		}
		n, err := strconv.Atoi(string(data[start:pos]))
		return n, err == nil && pos-start < 10
	}
	w, ok1 := tok()
	h, ok2 := tok()
	max, ok3 := tok()
	if !ok1 || !ok2 || !ok3 || max != 255 {
		return nil, errors.New("imaging: bad PGM header")
	}
	if int64(w)*int64(h) > maxPixels || w > 1<<16 || h > 1<<16 {
		return nil, errPixel
	}
	pos++
	if pos > len(data) || len(data)-pos < w*h {
		return nil, errors.New("imaging: short PGM")
	}
	b := imaging.NewBitmap(w, h, 8)
	copy(b.Pix, data[pos:pos+w*h])
	return b, nil
}

func (o *oracle) image(name string, data []byte) (string, string) {
	g, err := parsePGM(data)
	if err != nil {
		return "", class(err)
	}
	var b outbuf
	fmt.Fprintf(&b, "dims %d %d\n", g.Width, g.Height)
	fmt.Fprintf(&b, "gray8: %s\n", hashBitmap(imaging.FromImage(toGray(g))))
	imagingReport(&b, g)
	if o.rec != nil && (strings.HasPrefix(name, "line-") || strings.HasPrefix(name, "rnd-")) {
		recogReport(&b, o.rec, toGray(g))
	}
	return b.String(), ""
}

// rgba reads [w u32 LE][h u32 LE][w*h*4 bytes of premultiplied RGBA] as an image.RGBA, which
// imaging.FromImage and the recognizer convert with color.GrayModel (the Rust side: from_rgba8).
func (o *oracle) rgba(data []byte) (string, string) {
	if len(data) < 8 {
		return "", "imaging"
	}
	w, h := int64(binary.LittleEndian.Uint32(data)), int64(binary.LittleEndian.Uint32(data[4:]))
	if w*h > maxPixels {
		return "", errPixel.Error()
	}
	if int64(len(data)-8) != w*h*4 {
		return "", "imaging"
	}
	img := &image.RGBA{Pix: data[8:], Stride: int(w) * 4, Rect: image.Rect(0, 0, int(w), int(h))}
	g := imaging.FromImage(img)
	var b outbuf
	fmt.Fprintf(&b, "dims %d %d\nrgba: %s\n", g.Width, g.Height, hashBitmap(g))
	imagingReport(&b, g)
	if o.rec != nil {
		recogReport(&b, o.rec, img)
	}
	return b.String(), ""
}

func rect(x0, y0, x1, y1 int) image.Rectangle { return image.Rect(x0, y0, x1, y1) }

func compsHash(cs []imaging.Component) string {
	f := newFnv()
	for _, c := range cs {
		f.u32(uint32(c.Bounds.Min.X))
		f.u32(uint32(c.Bounds.Min.Y))
		f.u32(uint32(c.Bounds.Max.X))
		f.u32(uint32(c.Bounds.Max.Y))
		f.u32(uint32(c.Pixels))
	}
	return fmt.Sprintf("%d %016x", len(cs), f.h)
}

func imagingReport(b *outbuf, g *imaging.Bitmap) {
	w, h := g.Width, g.Height
	var bin, eroded *imaging.Bitmap
	step(b, "otsu_threshold", func() (string, error) { return fmt.Sprint(imaging.OtsuThreshold(g)), nil })
	step(b, "otsu", bm(func() *imaging.Bitmap { bin = imaging.Otsu(g); return bin }))
	step(b, "sauvola_15_034", bm(func() *imaging.Bitmap { return imaging.Sauvola(g, 15, 0.34) }))
	step(b, "sauvola_5_02", bm(func() *imaging.Bitmap { return imaging.Sauvola(g, 5, 0.2) }))
	step(b, "sauvola_3", bm(func() *imaging.Bitmap { return imaging.Sauvola(g, 3, 0.2) }))
	// Out-of-range float to int conversions: MinInt64 on amd64.
	step(b, "sauvola_5_neg1e20", bm(func() *imaging.Bitmap { return imaging.Sauvola(g, 5, -1e20) }))
	step(b, "sauvola_5_nan", bm(func() *imaging.Bitmap { return imaging.Sauvola(g, 5, math.NaN()) }))
	if bin == nil {
		return
	}
	var ang float64
	step(b, "skew_angle", func() (string, error) {
		ang = imaging.SkewAngle(bin)
		return fmt.Sprintf("%016x", f64(ang)), nil
	})
	step(b, "rotate_skew", bm(func() *imaging.Bitmap { return imaging.Rotate(bin, ang) }))
	step(b, "rotate_005", bm(func() *imaging.Bitmap { return imaging.Rotate(bin, 0.05) }))
	step(b, "rotate_gray", bm(func() *imaging.Bitmap { return imaging.Rotate(g, 0.05) }))
	for _, c := range []struct {
		name string
		a    float64
	}{{"nan", math.NaN()}, {"inf", math.Inf(1)}, {"1e39", 1e39}} {
		step(b, "rotate_"+c.name, bm(func() *imaging.Bitmap { return imaging.Rotate(bin, c.a) }))
	}
	// An angle in every octant of sin and cos (the port's j = 1..7 branches), and 1e6 and 1e9 for
	// the argument reduction (the Payne-Hanek path from 2^29), which the small angles never reach.
	for _, c := range []struct {
		name string
		a    float64
	}{{"0.9", 0.9}, {"2", 2.0}, {"-2.5", -2.5}, {"3.5", 3.5}, {"5", 5.0}, {"pi2", math.Pi / 2}, {"1e6", 1e6}, {"1e9", 1e9}} {
		step(b, "rotate_a"+c.name, bm(func() *imaging.Bitmap { return imaging.Rotate(bin, c.a) }))
	}
	for _, f := range []float64{0.5, 1, 1.7, 2.5, 0, -1, 0.01, 1e-20, 1e-50} {
		if f > 1 && float64(w*h)*f*f > 1<<22 {
			continue
		}
		step(b, fmt.Sprintf("scale_%v", f), bm(func() *imaging.Bitmap { return imaging.ScaleGray(g, f) }))
	}
	// Go checks only factor <= 0 and returns a bitmap of negative size, without an error, for a
	// factor that makes the size overflow (NaN, +Inf, and 1e39 once it is a float32). The crate
	// reports `imaging`; so does the oracle, and counts it.
	for _, c := range []struct {
		name string
		f    float64
	}{{"NaN", math.NaN()}, {"Inf", math.Inf(1)}, {"1e39", 1e39}} {
		step(b, "scale_"+c.name, func() (string, error) {
			r := imaging.ScaleGray(g, c.f)
			if r.Width < 0 || r.Height < 0 {
				noteRewrite("scale_negative_size")
				return "", errors.New("imaging: ScaleGray factor out of range")
			}
			return hashBitmap(r), nil
		})
	}
	step(b, "scale_bin", bm(func() *imaging.Bitmap { return imaging.ScaleGray(bin, 0.5) }))
	step(b, "dilate_3x3", bm(func() *imaging.Bitmap { return imaging.Dilate(bin, 3, 3) }))
	step(b, "erode_3x3", bm(func() *imaging.Bitmap { eroded = imaging.Erode(bin, 3, 3); return eroded }))
	step(b, "open_5x1", bm(func() *imaging.Bitmap { return imaging.Open(bin, 5, 1) }))
	step(b, "close_5x1", bm(func() *imaging.Bitmap { return imaging.Close(bin, 5, 1) }))
	step(b, "dilate_gray", bm(func() *imaging.Bitmap { return imaging.Dilate(g, 3, 3) }))
	step(b, "dilate_0x1", bm(func() *imaging.Bitmap { return imaging.Dilate(bin, 0, 1) }))
	var comps []imaging.Component
	for _, c := range []int{4, 8, 6} {
		step(b, fmt.Sprintf("conn_comp_%d", c), func() (string, error) {
			cs := imaging.ConnComp(bin, c)
			if c == 8 {
				comps = cs
			}
			return compsHash(cs), nil
		})
	}
	step(b, "conn_comp_gray", func() (string, error) { return compsHash(imaging.ConnComp(g, 4)), nil })
	step(b, "distance", bm(func() *imaging.Bitmap { return imaging.DistanceFunction(bin) }))
	if eroded != nil {
		for _, c := range []int{4, 8, 6} {
			step(b, fmt.Sprintf("seed_fill_%d", c), bm(func() *imaging.Bitmap { return imaging.SeedFill(eroded, bin, c) }))
		}
		step(b, "seed_fill_mismatch", bm(func() *imaging.Bitmap { return imaging.SeedFill(eroded, imaging.NewBitmap(w+1, h, 1), 4) }))
	}
	ops := []struct {
		name string
		op   imaging.Op
	}{
		{"clear", imaging.OpClear}, {"notsrc", imaging.OpNotSrc}, {"xor", imaging.OpSrcXorDst},
		{"and", imaging.OpSrcAndDst}, {"src", imaging.OpSrc}, {"or", imaging.OpSrcOrDst}, {"set", imaging.OpSet},
	}
	for _, o := range ops {
		if eroded != nil {
			step(b, "rop_"+o.name, bm(func() *imaging.Bitmap {
				d := bin.Clone()
				imaging.RasterOp(d, rect(-3, -2, w/2+1, h/2+1), o.op, eroded, image.Pt(2, 1))
				return d
			}))
		}
		step(b, "rop8_"+o.name, bm(func() *imaging.Bitmap {
			d := g.Clone()
			imaging.RasterOp(d, rect(1, 1, w, h), o.op, g, image.Pt(0, 2))
			return d
		}))
	}
	step(b, "rop_depth_mismatch", bm(func() *imaging.Bitmap {
		d := bin.Clone()
		imaging.RasterOp(d, d.Bounds(), imaging.OpSrc, g, image.Pt(0, 0))
		return d
	}))
	step(b, "grid", func() (string, error) {
		gr := imaging.NewGrid(g.Bounds(), 8)
		for i, c := range comps {
			if i >= 500 {
				break
			}
			gr.Insert(i, c.Bounds)
		}
		f := newFnv()
		n := 0
		for _, q := range []image.Rectangle{
			rect(0, 0, w/2, h/2), rect(w/3, h/3, w, h), rect(-5, -5, 3, 3), rect(w-4, h-4, w+9, h+9), rect(0, 0, 0, 0),
		} {
			ids := gr.Query(q)
			n += len(ids)
			f.u32(uint32(len(ids)))
			for _, id := range ids {
				f.u32(uint32(id))
			}
		}
		return fmt.Sprintf("%d %016x", n, f.h), nil
	})
	step(b, "grid_bad_cell", func() (string, error) { imaging.NewGrid(g.Bounds(), 0); return "ok", nil })
}

func hex(s string) string { return fmt.Sprintf("%x", s) }

// lossy is what the Rust crate does with model strings that are not UTF-8: String::from_utf8_lossy
// replaces each maximal subpart of an ill-formed sequence with one U+FFFD. That is neither
// strings.ToValidUTF8 (one replacement for a whole run of invalid bytes) nor ranging over the runes
// (one per byte): FF FE gives two, E2 82 41 gives one and then the A. A string that changes is
// counted as the rewrite lossy_string.
func lossy(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	noteRewrite("lossy_string")
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		if c < 0x80 {
			b.WriteByte(c)
			i++
			continue
		}
		// The sequence's length and the range of its second byte (the rest are 80..BF).
		n, lo, hi := 0, byte(0x80), byte(0xBF)
		switch {
		case c >= 0xC2 && c <= 0xDF:
			n = 1
		case c == 0xE0:
			n, lo = 2, 0xA0
		case c >= 0xE1 && c <= 0xEC, c == 0xEE, c == 0xEF:
			n = 2
		case c == 0xED:
			n, hi = 2, 0x9F
		case c == 0xF0:
			n, lo = 3, 0x90
		case c >= 0xF1 && c <= 0xF3:
			n = 3
		case c == 0xF4:
			n, hi = 3, 0x8F
		}
		j := i + 1
		ok := n > 0
		for k := 0; ok && k < n; k++ {
			if j >= len(s) || s[j] < lo || s[j] > hi {
				ok = false
				break
			}
			lo, hi = 0x80, 0xBF
			j++
		}
		if ok {
			b.WriteString(s[i:j])
		} else {
			b.WriteString("\uFFFD")
		}
		i = j
	}
	return b.String()
}

// lossyNames decodes every layer name as the Rust crate does, so that Tree, which quotes a name
// with %q (strconv.Quote), prints what the crate's `quote` prints for the decoded name.
func lossyNames(l *tessdata.Layer) {
	l.Name = lossy(l.Name)
	for _, c := range l.Children {
		lossyNames(c)
	}
}

func toGray(g *imaging.Bitmap) *image.Gray {
	return &image.Gray{Pix: g.Pix, Stride: g.Stride, Rect: image.Rect(0, 0, g.Width, g.Height)}
}

func scoredLines(b io.Writer, name string, s []recog.Scored) {
	fmt.Fprintf(b, "%s: %d\n", name, len(s))
	for _, x := range s {
		fmt.Fprintf(b, "  id=%d text=%s s=%d e=%d c=%016x r=%016x\n", x.UnicharID, hex(x.Text), x.Start, x.End, f64(x.Certainty), f64(x.Rating))
	}
}

func recogReport(b *outbuf, r *recog.Recognizer, img image.Image) {
	var out *nn.Tensor
	step(b, "forward", func() (string, error) {
		t, norm, err := r.Forward(img)
		if err != nil {
			return "", err
		}
		out = t
		f := newFnv()
		for i := 0; i < t.Map.Len(); i++ {
			for _, v := range t.Row(i) {
				f.u32(f32(v))
			}
		}
		return fmt.Sprintf("%d %d %016x scale=%016x black=%08x contrast=%08x", t.Map.Len(), t.Features, f.h,
			f64(norm.ScaleFactor), f32(norm.Black), f32(norm.Contrast)), nil
	})
	if out == nil {
		return
	}
	step(b, "greedy", func() (string, error) {
		syms, err := r.GreedyDecode(out)
		if err != nil {
			return "", err
		}
		var s strings.Builder
		for _, x := range syms {
			fmt.Fprintf(&s, " %d:%s:%d-%d", x.UnicharID, hex(x.Text), x.Start, x.End)
		}
		return fmt.Sprintf("%d%s", len(syms), s.String()), nil
	})
	for _, lex := range []bool{true, false} {
		r.SetLexicon(lex)
		name := fmt.Sprintf("beam_lexicon_%v", lex)
		childStep(name)
		s, err := safely(func() (string, error) {
			sc, err := r.BeamDecode(out)
			if err != nil {
				return "", err
			}
			var sb strings.Builder
			scoredLines(&sb, "", sc)
			return sb.String(), nil
		})
		if err != nil {
			fmt.Fprintf(b, "%s: ERR %s\n", name, class(err))
		} else {
			fmt.Fprintf(b, "%s%s", name, s)
		}
		if lex {
			step(b, "recognize", func() (string, error) {
				l, err := r.Recognize(img)
				if err != nil {
					return "", err
				}
				var s strings.Builder
				fmt.Fprintf(&s, "text=%s conf=%016x bounds=%v\n", hex(l.Text), f64(l.Confidence), l.Bounds)
				for _, w := range l.Words {
					fmt.Fprintf(&s, "  word=%s bounds=%v conf=%016x\n", hex(w.Text), w.Bounds, f64(w.Confidence))
				}
				return s.String(), nil
			})
		}
	}
	r.SetLexicon(true)
}

// vectors runs the tabulated activations and softmax over f64 values.
func (o *oracle) vectors(data []byte) (string, string) {
	if len(data) == 0 || len(data)%8 != 0 || len(data) > 8*4096 {
		return "", "bad_input"
	}
	v := make([]float64, len(data)/8)
	for i := range v {
		v[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
	}
	var b strings.Builder
	h := func(v []float64) string {
		f := newFnv()
		for _, x := range v {
			f.u64(f64(x))
		}
		return fmt.Sprintf("%016x", f.h)
	}
	// Go's tabulated tanh and logistic index their table with int(NaN), a negative index, and panic;
	// the crate returns NaN. A call that panics gives NaN here, and the input counts once as the
	// rewrite table_nan_panic.
	fired := false
	safe := func(f func(float64) float64) func(float64) float64 {
		return func(x float64) (r float64) {
			defer func() {
				if recover() != nil {
					fired, r = true, math.NaN()
				}
			}()
			return f(x)
		}
	}
	tanh, logistic := safe(nn.Tanh), safe(nn.Logistic)
	t, l, s := append([]float64(nil), v...), append([]float64(nil), v...), append([]float64(nil), v...)
	hasNaN := false
	for _, x := range v {
		hasNaN = hasNaN || math.IsNaN(x)
	}
	if hasNaN {
		for i, x := range v {
			t[i], l[i] = tanh(x), logistic(x)
		}
	} else {
		nn.TanhInPlace(t)
		nn.LogisticInPlace(l)
	}
	nn.SoftmaxInPlace(s)
	fmt.Fprintf(&b, "tanh %s\nlogistic %s\nsoftmax %s\n", h(t), h(l), h(s))
	for i := 0; i < len(v) && i < 8; i++ {
		fmt.Fprintf(&b, "%016x tanh=%016x logistic=%016x\n", f64(v[i]), f64(tanh(v[i])), f64(logistic(v[i])))
	}
	if fired {
		noteRewrite("table_nan_panic")
	}
	return b.String(), ""
}

func (o *oracle) unicharset(data []byte) (string, string) {
	u, err := tessdata.ParseUnicharset(data)
	if err != nil {
		return "", class(err)
	}
	f := newFnv()
	for i := 0; i < u.Size(); i++ {
		c, _ := u.Char(i)
		f.str(c.Text)
		f.str(c.Normed)
		f.str(c.Script)
		f.u32(c.Properties)
		f.str(u.Text(i))
		f.str(u.Normed(i))
		if u.IsSpaceDelimited(i) {
			f.u8(1)
		} else {
			f.u8(0)
		}
	}
	return fmt.Sprintf("size %d %016x\n", u.Size(), f.h), ""
}

func (o *oracle) recoder(data []byte, swap bool) (string, string) {
	rc, err := tessdata.ParseRecoder(data, swap)
	if err != nil {
		return "", class(err)
	}
	// Go allocates one byte per code value (validStart) without a limit; the Rust crate charges it
	// against max_alloc, 1 GiB here.
	if rc.CodeRange() > 1<<30 {
		noteRewrite("recoder_code_range")
		return "", "alloc_limit"
	}
	f := newFnv()
	for i := 0; i < rc.Size(); i++ {
		code := rc.Encode(i)
		f.u32(uint32(len(code)))
		for _, c := range code {
			f.u32(uint32(c))
		}
		if id, ok := rc.DecodeUnichar(code); ok {
			f.u32(uint32(id))
			f.u8(1)
		}
	}
	for c := int32(-1); c <= int32(rc.CodeRange()); c++ {
		if rc.IsValidFirstCode(c) {
			f.u8(1)
		} else {
			f.u8(0)
		}
	}
	return fmt.Sprintf("size %d range %d maxlen %d %016x\n", rc.Size(), rc.CodeRange(), rc.MaxCodeLen(), f.h), ""
}

// probeWords are looked up in each dawg, as unichar id sequences of the unedited model's
// unicharset (a word with a character the charset lacks is dropped).
var probeWords = []string{"the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "hello", "world",
	"Tesseract", "recognition", "dictionary", "cadmus", "12", "1999", "a", "I", "zzzz", "xqzv"}

func (o *oracle) wordIDs() [][]int {
	c, err := tessdata.ParseContainer(o.model)
	if err != nil {
		return nil
	}
	e, _ := c.Entry(tessdata.TypeLSTMUnicharset)
	u, err := tessdata.ParseUnicharset(e)
	if err != nil {
		return nil
	}
	ids := map[string]int{}
	for id := 0; id < u.Size(); id++ {
		if _, ok := ids[u.Text(id)]; !ok {
			ids[u.Text(id)] = id
		}
	}
	var out [][]int
	for _, w := range probeWords {
		var p []int
		for _, r := range w {
			id, ok := ids[string(r)]
			if !ok {
				p = nil
				break
			}
			p = append(p, id)
		}
		if p != nil {
			out = append(out, p)
		}
	}
	return out
}

func (o *oracle) dawg(data []byte, swap bool) (string, string) {
	d, err := tessdata.ParseDawg(data, swap)
	if err != nil {
		return "", class(err)
	}
	f := newFnv()
	n := d.NumEdges()
	for e := 0; e < n; e++ {
		f.u32(uint32(d.UnicharID(int64(e))))
		f.u64(uint64(d.NextNode(int64(e))))
		if d.EndOfWord(int64(e)) {
			f.u8(1)
		} else {
			f.u8(0)
		}
	}
	probes := [][]int{{1}, {1, 2}, {2, 3, 4}, {5}, {10, 11}, {}}
	for _, p := range o.wordIDs() {
		probes = append(probes, p, p[:len(p)-1])
	}
	for _, p := range probes {
		if d.Contains(p) {
			f.u8(1)
		} else {
			f.u8(0)
		}
		if d.HasPrefix(p) {
			f.u8(1)
		} else {
			f.u8(0)
		}
	}
	for node := 0; node < n; node += 7 {
		for id := 0; id < 8; id++ {
			f.u64(uint64(d.EdgeChar(int64(node), id, false)))
			f.u64(uint64(d.EdgeChar(int64(node), id, true)))
		}
	}
	return fmt.Sprintf("edges %d size %d %016x\n", n, d.UnicharsetSize, f.h), ""
}

func summary(b *strings.Builder, m *tessdata.Model) {
	rec := m.Recognizer
	fmt.Fprintf(b, "version %s swapped %v\n", hex(lossy(m.Version)), m.Swapped)
	fmt.Fprintf(b, "spec %s flags %d iter %d sample %d null %d adam %08x lr %08x mom %08x\n", hex(lossy(rec.NetworkStr)),
		rec.TrainingFlags, rec.TrainingIteration, rec.SampleIteration, rec.NullChar,
		f32(rec.AdamBeta), f32(rec.LearningRate), f32(rec.Momentum))
	fmt.Fprintf(b, "unicharset %d recoder %d range %d maxlen %d\n", m.Unicharset.Size(), m.Recoder.Size(), m.Recoder.CodeRange(), m.Recoder.MaxCodeLen())
	for _, d := range []*tessdata.Dawg{m.PuncDawg, m.WordDawg, m.NumberDawg} {
		if d == nil {
			b.WriteString("dawg -\n")
		} else {
			fmt.Fprintf(b, "dawg %d\n", d.NumEdges())
		}
	}
	lossyNames(rec.Network)
	rec.Network.Tree(b)
}
