package main

import (
	"bytes"
	"encoding/binary"
	hexenc "encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unicode/utf8"

	"github.com/dobbo-ca/cadmus/internal/imaging"
	"github.com/dobbo-ca/cadmus/internal/recog"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// A .mut file is a recipe of text lines:
//
//	part <uni|rec|dawgp|dawgw|dawgn|lstm> [swap]   optional: edit and parse this component alone
//	trunc N                                       keep the first N bytes
//	flip OFFSET BIT                               xor 1<<BIT into the byte (ignored past the end)
//	set OFFSET VALUE                              set the byte (ignored past the end)
//	setf OFFSET HEXBITS                           set 8 bytes to the bits of an f64 (ignored past the end)
//	data HEX                                      replace everything with these bytes
//
// Without `part` the edits apply to the whole model, which is then loaded as a model and as a
// recognizer, and run over a fixed synthetic line.
var partTypes = map[string]tessdata.Type{
	"uni": tessdata.TypeLSTMUnicharset, "rec": tessdata.TypeLSTMRecoder, "lstm": tessdata.TypeLSTM,
	"dawgp": tessdata.TypeLSTMPuncDawg, "dawgw": tessdata.TypeLSTMSystemDawg, "dawgn": tessdata.TypeLSTMNumberDawg,
}

func applyEdits(base []byte, lines []string) []byte {
	d := append([]byte(nil), base...)
	for _, l := range lines {
		var a, b int
		switch {
		case strings.HasPrefix(l, "trunc "):
			fmt.Sscanf(l, "trunc %d", &a)
			if a < len(d) {
				d = d[:a]
			}
		case strings.HasPrefix(l, "flip "):
			fmt.Sscanf(l, "flip %d %d", &a, &b)
			if a < len(d) {
				d[a] ^= 1 << uint(b)
			}
		case strings.HasPrefix(l, "set "):
			fmt.Sscanf(l, "set %d %d", &a, &b)
			if a < len(d) {
				d[a] = byte(b)
			}
		case strings.HasPrefix(l, "setf "):
			var bits uint64
			fmt.Sscanf(l, "setf %d %x", &a, &bits)
			if a+8 <= len(d) {
				binary.LittleEndian.PutUint64(d[a:], bits)
			}
		case strings.HasPrefix(l, "data "):
			d, _ = hexenc.DecodeString(strings.TrimPrefix(l, "data "))
		}
	}
	return d
}

// layerDepth is the number of layers between the root and its deepest descendant.
func layerDepth(l *tessdata.Layer) int {
	d := 0
	for _, c := range l.Children {
		d = max(d, 1+layerDepth(c))
	}
	return d
}

// noiseLine is the fixed synthetic image the mutated models read: a gradient with xorshift noise.
func noiseLine() *imaging.Bitmap {
	const w, h = 120, 36
	b := imaging.NewBitmap(w, h, 8)
	x := xs{88172645463325252}
	for i := range b.Pix {
		b.Pix[i] = byte((i%w)*2+int(x.next()%64)) ^ byte(i/w*3)
	}
	return b
}

func (o *oracle) mutant(recipe []byte) (string, string) {
	if o.model == nil {
		return "", "no_model"
	}
	lines := strings.Split(strings.TrimSpace(string(recipe)), "\n")
	b := outbuf{live: childLive}
	if strings.HasPrefix(lines[0], "part ") {
		f := strings.Fields(lines[0])
		swap := len(f) > 2 && f[2] == "swap"
		t := partTypes[f[1]]
		c, err := tessdata.ParseContainer(o.model)
		if err != nil {
			return "", class(err)
		}
		base, _ := c.Entry(t)
		d := applyEdits(base, lines[1:])
		switch f[1] {
		case "uni":
			if !utf8.Valid(d) { // the Rust crate requires a UTF-8 unicharset
				noteRewrite("unicharset_not_utf8")
				return "", "tessdata"
			}
			return o.unicharset(d)
		case "rec":
			return o.recoder(d, swap)
		case "lstm":
			s, err := safely(func() (string, error) {
				rec, err := tessdata.ParseRecognizer(d, swap)
				if err != nil {
					return "", err
				}
				// The crate allows a network 64 layers deep at most; Go has no such limit.
				if layerDepth(rec.Network) > 64 {
					noteRewrite("layer_nesting")
					return "", errors.New("tessdata: network nested deeper than 64 layers")
				}
				var sb strings.Builder
				fmt.Fprintf(&sb, "spec %s flags %d iter %d sample %d null %d adam %08x lr %08x mom %08x\n", hex(lossy(rec.NetworkStr)),
					rec.TrainingFlags, rec.TrainingIteration, rec.SampleIteration, rec.NullChar,
					f32(rec.AdamBeta), f32(rec.LearningRate), f32(rec.Momentum))
				lossyNames(rec.Network)
				rec.Network.Tree(&sb)
				return sb.String(), nil
			})
			if err != nil {
				return "", class(err)
			}
			return s, ""
		default:
			return o.dawg(d, swap)
		}
	}
	m := applyEdits(o.model, lines)
	if len(m) > maxModelBytes {
		return "", errModel.Error()
	}
	// The Rust crate rejects a unicharset that is not UTF-8 (Go accepts it); mirror that.
	utf8Bad := false
	if mc, err := tessdata.ParseContainer(m); err == nil {
		if u, ok := mc.Entry(tessdata.TypeLSTMUnicharset); ok && !utf8.Valid(u) {
			utf8Bad = true
		}
	}
	childStep("load")
	s, err := safely(func() (string, error) {
		model, err := tessdata.LoadModel(m)
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		summary(&sb, model)
		return sb.String(), nil
	})
	if err == nil && utf8Bad {
		noteRewrite("unicharset_not_utf8")
		err = errors.New("tessdata: unicharset is not UTF-8")
	}
	if err != nil {
		fmt.Fprintf(&b, "load: ERR %s\n", class(err))
	} else {
		fmt.Fprintf(&b, "load: ok\n%s", s)
	}
	var r *recog.Recognizer
	childStep("recognizer")
	_, err = safely(func() (string, error) {
		var err error
		r, err = recog.NewRecognizer(m)
		return "", err
	})
	if err == nil && utf8Bad {
		noteRewrite("unicharset_not_utf8")
		err = errors.New("recog: unicharset is not UTF-8")
	}
	if err != nil {
		fmt.Fprintf(&b, "recognizer: ERR %s\n", class(err))
		return b.String(), ""
	}
	b.WriteString("recognizer: ok\n")
	recogReport(&b, r, toGray(noiseLine()))
	return b.String(), ""
}

// mutantChild runs a recipe in a child process with an address-space limit of 3 GiB (a Go process
// needs far more address space than it uses, so a tighter limit fails sporadically). An edited model can declare a tensor of terabytes, and Go has no limit of
// its own: its allocation is a fatal error, not a panic, and an allocation that fits would take
// the machine down. A child that dies of memory has its report up to the step it died in, and that
// step is reported as `alloc_limit`, the class the Rust crate's Limits give such a model.
func (o *oracle) mutantChild(name string) (string, string) {
	cmd := exec.Command(os.Args[0], "-model", *modelPath, "-one", filepath.Join(o.dir, name))
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if os.Getenv("ORACLE_MSG") != "" {
		os.Stderr.Write(stderr.Bytes())
	}
	for _, l := range strings.Split(stderr.String(), "\n") {
		if k, ok := strings.CutPrefix(l, "REWRITE "); ok {
			addRewrite(k)
		}
	}
	s := stdout.String()
	if i := strings.IndexByte(s, 0); i >= 0 {
		return "", strings.TrimSpace(s[i+1:])
	}
	if err != nil {
		if !strings.Contains(stderr.String(), "out of memory") && !strings.Contains(stderr.String(), "cannot allocate") {
			return "", "crash"
		}
		// The step the child died in is the last STEP line (none for a component recipe); the
		// Rust crate's Limits refuse it.
		step := ""
		for _, l := range strings.Split(stderr.String(), "\n") {
			if strings.HasPrefix(l, "STEP ") {
				step = l[5:]
			}
		}
		noteRewrite("out_of_memory")
		switch step {
		case "":
			return "", "alloc_limit"
		case "load":
			s += "load: ERR alloc_limit\nrecognizer: ERR alloc_limit\n"
		default:
			s += step + ": ERR alloc_limit\n"
		}
	}
	// A NaN that reaches the tabulated tanh or logistic indexes the table with int(NaN), a negative
	// index: a Go runtime panic in the forward pass. The crate fails the forward pass there, and
	// the recognizer reports it as `recog`. A NaN that reaches only the softmax gives output in
	// both, so it needs no rewrite.
	if strings.Contains(stderr.String(), "FWDERR runtime error: index out of range [-9223372036854775808]") &&
		strings.Contains(s, "forward: ERR other\n") {
		noteRewrite("forward_nan_in_table")
		s = strings.Replace(s, "forward: ERR other\n", "forward: ERR recog\n", 1)
	}
	// A recognizer that Go builds but whose forward pass dies of a Go runtime panic (a negative
	// size in an edited layer, say) is one the Rust crate rejects when it builds it. Any other
	// forward error stays as Go reported it.
	if i := strings.Index(s, "recognizer: ok\n"); i >= 0 && strings.Contains(s[i:], "forward: ERR other\n") &&
		strings.Contains(stderr.String(), "FWDERR runtime error:") {
		noteRewrite("forward_runtime_error")
		s = s[:i] + "recognizer: ERR recog\n"
	}
	return s, ""
}

// runOne is the child side of mutantChild.
func runOne(path string, model []byte) {
	lim := &syscall.Rlimit{Cur: 3 << 30, Max: 3 << 30}
	if err := syscall.Setrlimit(syscall.RLIMIT_AS, lim); err != nil {
		fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fatal(err)
	}
	inChild = true
	o := &oracle{model: model}
	if !bytes.HasPrefix(data, []byte("part ")) {
		childLive = os.Stdout // the whole-model report is streamed as it grows
	}
	report, class := o.mutant(data)
	switch {
	case class != "":
		fmt.Print("\x00" + class + "\n")
	case childLive == nil:
		fmt.Print(report)
	}
}
