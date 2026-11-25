package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// xs is xorshift64, so the inputs do not depend on the Go version's math/rand.
type xs struct{ s uint64 }

func (x *xs) next() uint64 {
	x.s ^= x.s << 13
	x.s ^= x.s >> 7
	x.s ^= x.s << 17
	return x.s
}
func (x *xs) intn(n int) int { return int(x.next() % uint64(n)) }
func (x *xs) unit() float64  { return float64(x.next()>>11) / (1 << 53) }

func pgm(w, h int, pix []byte) []byte {
	return append([]byte(fmt.Sprintf("P5\n%d %d\n255\n", w, h)), pix...)
}

func put(dir, name string, data []byte) { write(dir, name, string(data)) }

// randImage is a small gray image of one of five kinds: noise, constant, gradient, dark blobs on
// a light page, and bars like text rows. Sizes include 1 pixel and non-multiples of 8.
func randImage(x *xs) (int, int, []byte) {
	dims := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 40, 64, 80}
	w, h := dims[x.intn(len(dims))], dims[x.intn(len(dims))]
	p := make([]byte, w*h)
	switch x.intn(5) {
	case 0:
		for i := range p {
			p[i] = byte(x.next())
		}
	case 1:
		v := []byte{0, 128, 255}[x.intn(3)]
		for i := range p {
			p[i] = v
		}
	case 2:
		a, b := x.intn(9), x.intn(9)
		for i := range p {
			p[i] = byte((i%w)*a + (i/w)*b)
		}
	case 3:
		for i := range p {
			p[i] = 220
		}
		for n := x.intn(8); n >= 0; n-- {
			x0, y0 := x.intn(w), x.intn(h)
			x1, y1 := x0+1+x.intn(w), y0+1+x.intn(h)
			v := byte(x.intn(100))
			for y := y0; y < min(y1, h); y++ {
				for xx := x0; xx < min(x1, w); xx++ {
					p[y*w+xx] = v
				}
			}
		}
	default:
		for i := range p {
			p[i] = 235
			if (i/w)%6 < 3 && x.intn(4) != 0 {
				p[i] = byte(x.intn(90))
			}
		}
	}
	return w, h, p
}

func vectorBytes(x *xs) []byte {
	n := 1 + x.intn(64)
	scale := []float64{1, 4, 100, 1e3, 1e6, 1e12, 1e300}[x.intn(7)]
	special := []float64{0, math.Copysign(0, -1), 4095.0 / 256, -4095.0 / 256, 1.0 / 256, 15.9999, 16, -16, 1e-300, 5e-324, 86, -86, 87, math.Inf(1), math.Inf(-1), math.NaN()}
	out := make([]byte, 8*n)
	for i := 0; i < n; i++ {
		v := (x.unit()*2 - 1) * scale
		if i%8 == 3 {
			v = special[x.intn(len(special))]
		}
		binary.LittleEndian.PutUint64(out[8*i:], math.Float64bits(v))
	}
	return out
}

func generate(testdata, out string, model []byte) {
	x := &xs{0x9e3779b97f4a7c15}
	count := 0
	emit := func(name string, data []byte) { put(out, name, data); count++ }

	// Line images: each as an imaging input and, `line-` prefixed, a recognizer input; some with
	// noise or half the columns cut.
	for _, set := range []string{"native", "h36"} {
		files, _ := filepath.Glob(filepath.Join(testdata, "lines", set, "*.pgm"))
		sort.Strings(files)
		for i, f := range files {
			data, err := os.ReadFile(f)
			if err != nil {
				fatal(err)
			}
			g, err := parsePGM(data)
			if err != nil {
				fatal(err)
			}
			id := strings.TrimSuffix(filepath.Base(f), ".pgm")
			name := "line-" + set + "-" + id
			emit(name+".pgm", data)
			if i%4 == 0 {
				p := append([]byte(nil), g.Pix...)
				for n := 0; n < 40; n++ {
					p[x.intn(len(p))] = byte(x.next())
				}
				emit(name+"~n.pgm", pgm(g.Width, g.Height, p))
			}
			if i%8 == 1 {
				w2 := g.Width / 2
				p := make([]byte, 0, w2*g.Height)
				for y := 0; y < g.Height; y++ {
					p = append(p, g.Pix[y*g.Width:y*g.Width+w2]...)
				}
				emit(name+"~c.pgm", pgm(w2, g.Height, p))
			}
		}
	}

	// The Leptonica scan the goldens come from: whole, with flipped pixels, and a crop.
	if raw, err := os.ReadFile(filepath.Join(testdata, "golden", "gray.bin")); err == nil {
		w, h := int(binary.LittleEndian.Uint32(raw)), int(binary.LittleEndian.Uint32(raw[4:]))
		pix := raw[12 : 12+w*h]
		emit("golden-gray.pgm", pgm(w, h, pix))
		p := append([]byte(nil), pix...)
		for n := 0; n < 5000; n++ {
			p[x.intn(len(p))] = byte(x.next())
		}
		emit("golden-gray~n.pgm", pgm(w, h, p))
		c := make([]byte, 0, 300*300)
		for y := 100; y < 400; y++ {
			c = append(c, pix[y*w+200:y*w+500]...)
		}
		emit("golden-gray~c.pgm", pgm(300, 300, c))
	} else {
		fatal(err)
	}

	for i := 0; i < 400; i++ {
		w, h, p := randImage(x)
		emit(fmt.Sprintf("rnd-%04d.pgm", i), pgm(w, h, p))
	}
	for i := 0; i < 150; i++ {
		emit(fmt.Sprintf("vec-%04d.nn", i), vectorBytes(x))
	}

	if model == nil {
		fatal(fmt.Errorf("-gen needs -model"))
	}
	c, err := tessdata.ParseContainer(model)
	if err != nil {
		fatal(err)
	}
	// Recipes. Whole model: cuts, edits of the offset table, edits anywhere, edits inside the lstm
	// component's first bytes (the spec string and layer headers).
	n := 0
	recipe := func(part string, lines ...string) {
		head := ""
		if part != "" {
			head = "part " + part + "\n"
		}
		tag := part
		if tag == "" {
			tag = "model"
		}
		tag = strings.ReplaceAll(tag, " ", "-")
		emit(fmt.Sprintf("mut-%s-%04d.mut", tag, n), []byte(head+strings.Join(lines, "\n")+"\n"))
		n++
	}
	edits := func(part string, size, hdr int, truncs []int, nflip, nset int) {
		for _, t := range truncs {
			recipe(part, fmt.Sprintf("trunc %d", t))
		}
		pos := func() int {
			if x.intn(10) < 7 {
				return x.intn(min(hdr, size))
			}
			return x.intn(size)
		}
		for i := 0; i < nflip; i++ {
			recipe(part, fmt.Sprintf("flip %d %d", pos(), x.intn(8)))
		}
		for i := 0; i < nset; i++ {
			recipe(part, fmt.Sprintf("set %d %d", pos(), []int{0, 255, 1, 128, x.intn(256)}[x.intn(5)]))
		}
	}
	L := len(model)
	lstm, _ := c.Entry(tessdata.TypeLSTM)
	lstmOff := bytes.Index(model, lstm[:64])
	edits("", L, 200, []int{0, 1, 3, 4, 5, 100, 195, 196, 200, 1000, L / 4, L / 2, L * 3 / 4, L - 1, L - 100}, 40, 20)
	for i := 0; i < 30; i++ {
		recipe("", fmt.Sprintf("flip %d %d", x.intn(L), x.intn(8)))
	}
	for i := 0; i < 20; i++ {
		recipe("", fmt.Sprintf("set %d %d", lstmOff+x.intn(300), x.intn(256)))
	}
	for _, p := range []struct {
		name string
		t    tessdata.Type
		hdr  int
	}{
		{"uni", tessdata.TypeLSTMUnicharset, 256}, {"rec", tessdata.TypeLSTMRecoder, 64},
		{"dawgp", tessdata.TypeLSTMPuncDawg, 40}, {"dawgw", tessdata.TypeLSTMSystemDawg, 40},
		{"dawgn", tessdata.TypeLSTMNumberDawg, 40}, {"lstm", tessdata.TypeLSTM, 400},
	} {
		b, _ := c.Entry(p.t)
		size := len(b)
		recipe(p.name, "trunc "+fmt.Sprint(size))
		recipe(p.name+" swap", "trunc "+fmt.Sprint(size))
		edits(p.name, size, p.hdr, []int{0, 1, 2, 3, 4, 8, 16, size / 4, size / 2, size * 3 / 4, size - 16, size - 8, size - 1}, 40, 15)
		for i := 0; i < 10; i++ {
			recipe(p.name+" swap", fmt.Sprintf("flip %d %d", x.intn(min(p.hdr, size)), x.intn(8)))
		}
	}
	generateRGBA(testdata, x, emit)
	generateHardening(model, c, emit)
	fmt.Fprintf(os.Stderr, "%d inputs\n", count)
}

func rgba(w, h int, pix []byte) []byte {
	b := binary.LittleEndian.AppendUint32(nil, uint32(w))
	return append(binary.LittleEndian.AppendUint32(b, uint32(h)), pix...)
}

// generateRGBA writes `rgba-*.rgba` inputs: 24 small images with random premultiplied pixels (the
// colour channels never above the alpha, alpha anywhere in 0..255, some opaque) and a few more
// whose colour channels are above the alpha (image.RGBA does not forbid them and GrayModel does
// not look at the alpha), and the first three line images tinted and made translucent, which the
// recognizer reads.
func generateRGBA(testdata string, x *xs, emit func(string, []byte)) {
	dims := []int{1, 2, 3, 5, 8, 17, 40, 64}
	for i := 0; i < 24; i++ {
		w, h := dims[x.intn(len(dims))], dims[x.intn(len(dims))]
		p := make([]byte, 4*w*h)
		for j := 0; j < len(p); j += 4 {
			a := byte(x.next())
			if x.intn(4) == 0 {
				a = 255
			}
			for k := 0; k < 3; k++ {
				c := byte(x.next())
				if i < 20 {
					c = byte(int(c) * int(a) / 255)
				}
				p[j+k] = c
			}
			p[j+3] = a
		}
		emit(fmt.Sprintf("rgba-%04d.rgba", i), rgba(w, h, p))
	}
	files, _ := filepath.Glob(filepath.Join(testdata, "lines", "h36", "*.pgm"))
	sort.Strings(files)
	for i, f := range files[:3] {
		data, err := os.ReadFile(f)
		if err != nil {
			fatal(err)
		}
		g, err := parsePGM(data)
		if err != nil {
			fatal(err)
		}
		p := make([]byte, 0, 4*len(g.Pix))
		for _, v := range g.Pix {
			a := 150 + x.intn(106)
			p = append(p, byte(int(v)*a/255), byte(int(v)*3/4*a/255), byte(int(v)/2*a/255), byte(a))
		}
		emit(fmt.Sprintf("rgba-line-%d.rgba", i), rgba(g.Width, g.Height, p))
	}
}

// nested is an lstm component whose network is n Series layers around one Convolve, followed by
// the recognizer fields: Go loads it at any depth.
func nested(n int) []byte {
	var b []byte
	u32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	header := func(tag byte, plumbing bool) {
		b = append(b, tag, 2, 0) // type tag, training state (disabled), needs_to_backprop
		u32(0)                   // flags
		u32(1)                   // ni
		u32(1)                   // no
		u32(0)                   // num_weights
		u32(0)                   // name length
		if plumbing {
			u32(1) // stack size
		}
	}
	for i := 0; i < n; i++ {
		header(byte(tessdata.LayerSeries), true)
	}
	header(byte(tessdata.LayerConvolve), false)
	u32(0)  // half_x
	u32(0)  // half_y
	u32(0)  // spec string
	u32(64) // training_flags: TF_COMPRESS_UNICHARSET
	u32(0)  // training_iteration
	u32(0)  // sample_iteration
	u32(0)  // null_char
	for i := 0; i < 3; i++ {
		u32(0) // adam_beta, learning_rate, momentum
	}
	return b
}

// matrixOffset is where the values of m start in the model: the matrix is int32 rows, int32
// cols, one f64 padding element, then the values.
func matrixOffset(model []byte, m *tessdata.Matrix) int {
	key := binary.LittleEndian.AppendUint32(nil, uint32(m.Rows))
	key = binary.LittleEndian.AppendUint32(key, uint32(m.Cols))
	for from := 0; ; {
		i := bytes.Index(model[from:], key)
		if i < 0 {
			fatal(fmt.Errorf("matrix %dx%d not found in the model", m.Rows, m.Cols))
		}
		i += from
		ok := i+16+8*8 <= len(model)
		for j := 0; ok && j < 8 && j < len(m.Values); j++ {
			ok = binary.LittleEndian.Uint64(model[i+16+8*j:]) == math.Float64bits(m.Values[j])
		}
		if ok {
			return i + 16
		}
		from = i + 1
	}
}

func firstLayer(l *tessdata.Layer, t tessdata.LayerType) *tessdata.Layer {
	if l.Type == t && len(l.Matrices) > 0 {
		return l
	}
	for _, c := range l.Children {
		if f := firstLayer(c, t); f != nil {
			return f
		}
	}
	return nil
}

// generateHardening writes the inputs for the places where the crate is stricter than Go or
// reports a Go panic as an error (docs/parity.md, "Differences from Go that the oracle
// encodes"): infinite and NaN weights in a layer with a tabulated activation (tanh, logistic: Go
// panics in the forward pass) and in the softmax layer (Go gives output), a network nested 64,
// 65 and 500 layers deep, and a unicharset that is not UTF-8.
func generateHardening(model []byte, c *tessdata.Container, emit func(string, []byte)) {
	const nan, inf, ninf = "7ff8000000000000", "7ff0000000000000", "fff0000000000000"
	m, err := tessdata.LoadModel(model)
	if err != nil {
		fatal(err)
	}
	net := m.Recognizer.Network
	at := func(l *tessdata.Layer, matrix, row, col int) int {
		mx := &l.Matrices[matrix]
		return matrixOffset(model, mx) + 8*(row*mx.Cols+col)
	}
	setf := func(off int, bits string) string { return fmt.Sprintf("setf %d %s\n", off, bits) }
	recipe := func(name string, lines ...string) {
		emit("mut-float-"+name+".mut", []byte(strings.Join(lines, "")))
	}
	tanh := firstLayer(net, tessdata.LayerTanh)
	lstm := firstLayer(net, tessdata.LayerLSTM)
	soft := firstLayer(net, tessdata.LayerSoftmax)
	if tanh == nil || lstm == nil || soft == nil {
		fatal(errors.New("the model lacks a tanh, lstm or softmax layer"))
	}
	last := func(l *tessdata.Layer, matrix int) int { return l.Matrices[matrix].Cols - 1 }
	// Tanh layer: a NaN weight, an infinite pair that sums to NaN, and a lone infinity (no NaN).
	recipe("tanh-nan", setf(at(tanh, 0, 0, 0), nan))
	recipe("tanh-infpair", setf(at(tanh, 0, 0, 0), ninf), setf(at(tanh, 0, 0, last(tanh, 0)), inf))
	recipe("tanh-biasninf", setf(at(tanh, 0, 0, last(tanh, 0)), ninf))
	// LSTM: the CI gate goes through tanh, the GI gate through logistic.
	recipe("lstm-ci-nan", setf(at(lstm, 0, 0, 0), nan))
	recipe("lstm-gi-infpair", setf(at(lstm, 1, 0, 0), ninf), setf(at(lstm, 1, 0, last(lstm, 1)), inf))
	recipe("lstm-gi-biasinf", setf(at(lstm, 1, 0, last(lstm, 1)), inf))
	// Softmax layer, no table: Go gives output, so does the crate.
	recipe("softmax-nan-row0", setf(at(soft, 0, 0, 0), nan))
	recipe("softmax-nan-row5", setf(at(soft, 0, 5, 0), nan))
	recipe("softmax-infpair", setf(at(soft, 0, 0, 0), ninf), setf(at(soft, 0, 0, last(soft, 0)), inf))
	recipe("softmax-inf-row3", setf(at(soft, 0, 3, last(soft, 0)), inf))

	// Nesting: Go loads any depth, the crate stops after 64.
	for _, n := range []int{64, 65, 500} {
		emit(fmt.Sprintf("mut-nest-%d.mut", n), []byte(fmt.Sprintf("part lstm\ndata %x\n", nested(n))))
	}

	// A unicharset that is not UTF-8: the first letter z of a line gets the byte 0xFF.
	uni, _ := c.Entry(tessdata.TypeLSTMUnicharset)
	z := bytes.Index(uni, []byte("\nz "))
	if z < 0 {
		fatal(errors.New("no z in the unicharset"))
	}
	emit("mut-uni-notutf8.mut", []byte(fmt.Sprintf("part uni\nset %d 255\n", z+1)))
	emit("mut-model-uni-notutf8.mut", []byte(fmt.Sprintf("set %d 255\n", bytes.Index(model, uni[:64])+z+1)))

	generateStrings(model, c, m, tanh, emit)
}

// generateStrings writes the inputs for the strings the crate decodes lossily and quotes as Go's
// %q: a layer name, the spec string and the version. Each gets bytes that Go quotes as an escape
// (a control byte, DEL, a no-break space, a zero-width space), a combining mark, a variation
// selector and a Hangul filler that stay literal, and invalid UTF-8 that the crate replaces per
// maximal subpart (FF FE is two U+FFFD, E2 82 followed by a valid byte is one). The layer name and the spec are edited in the lstm component
// alone and in the whole model; the version only in the whole model.
func generateStrings(model []byte, c *tessdata.Container, m *tessdata.Model, layer *tessdata.Layer, emit func(string, []byte)) {
	lstm, _ := c.Entry(tessdata.TypeLSTM)
	lstmOff := bytes.Index(model, lstm[:64])
	version := c.Version()
	spec := m.Recognizer.NetworkStr
	if len(layer.Name) < 3 || len(spec) < 3 || len(version) < 3 {
		fatal(errors.New("the model's strings are too short to edit"))
	}
	// where returns the offset of the bytes of s, which the component stores after a 4-byte length.
	where := func(data []byte, s string, last bool) int {
		key := append(binary.LittleEndian.AppendUint32(nil, uint32(len(s))), s...)
		i := bytes.Index(data, key)
		if last {
			i = bytes.LastIndex(data, key)
		}
		if i < 0 {
			fatal(fmt.Errorf("string %q not found in the model", s))
		}
		return i + 4
	}
	nameAt := where(lstm, layer.Name, false)
	specAt := where(lstm, spec, true) // the trailer: the version also holds the spec text
	versionAt := bytes.Index(model, []byte(version))
	if versionAt < 0 {
		fatal(errors.New("version not found in the model"))
	}
	for _, e := range []struct {
		tag  string
		vals []byte
	}{
		{"ctl01", []byte{0x01}}, {"del7f", []byte{0x7f}}, {"nbsp", []byte{0xC2, 0xA0}},
		{"zwsp", []byte{0xE2, 0x80, 0x8B}}, {"comb", []byte{'a', 0xCC, 0x81}},
		{"fffe", []byte{0xFF, 0xFE}}, {"e282", []byte{0xE2, 0x82}},
		// Default-ignorable letters and marks, which Go prints literally: U+FE0F and U+3164.
		{"vs16", []byte{0xEF, 0xB8, 0x8F}}, {"hf", []byte{0xE3, 0x85, 0xA4}},
	} {
		sets := func(at int) string {
			var b strings.Builder
			for i, v := range e.vals {
				fmt.Fprintf(&b, "set %d %d\n", at+i, v)
			}
			return b.String()
		}
		emit("mut-str-part-name-"+e.tag+".mut", []byte("part lstm\n"+sets(nameAt)))
		emit("mut-str-part-spec-"+e.tag+".mut", []byte("part lstm\n"+sets(specAt)))
		emit("mut-str-model-name-"+e.tag+".mut", []byte(sets(lstmOff+nameAt)))
		emit("mut-str-model-spec-"+e.tag+".mut", []byte(sets(lstmOff+specAt)))
		emit("mut-str-model-version-"+e.tag+".mut", []byte(sets(versionAt)))
	}
}
