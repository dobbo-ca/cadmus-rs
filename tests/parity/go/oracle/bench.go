package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/dobbo-ca/cadmus/internal/imaging"
	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/recog"
)

// timeIt prints `name<TAB>ms`: the fastest of at least 3 runs (and up to 1 s of them) after a
// warm-up run. examples/bench.rs does the same.
func timeIt(name string, f func()) {
	f()
	best := time.Duration(1 << 62)
	start := time.Now()
	for n := 0; n < 3 || (time.Since(start) < time.Second && n < 200); n++ {
		t := time.Now()
		f()
		if d := time.Since(t); d < best {
			best = d
		}
	}
	fmt.Printf("%s\t%.3f\n", name, float64(best.Nanoseconds())/1e6)
}

func loadLines(dir string) []*imaging.Bitmap {
	files, _ := filepath.Glob(filepath.Join(dir, "*.pgm"))
	sort.Strings(files)
	var out []*imaging.Bitmap
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			fatal(err)
		}
		g, err := parsePGM(data)
		if err != nil {
			fatal(err)
		}
		out = append(out, g)
	}
	return out
}

// bench times the recognizer on the line corpus and the imaging operators on the Leptonica scan.
func bench(testdata string, model []byte) {
	r, err := recog.NewRecognizer(model)
	if err != nil {
		fatal(err)
	}
	for _, set := range []string{"native", "h36"} {
		lines := loadLines(filepath.Join(testdata, "lines", set))
		imgs := make([]*image.Gray, len(lines))
		outs := make([]*nn.Tensor, len(lines))
		for i, g := range lines {
			imgs[i] = toGray(g)
			if outs[i], _, err = r.Forward(imgs[i]); err != nil {
				fatal(err)
			}
		}
		timeIt("forward_"+set, func() {
			for _, im := range imgs {
				_, _, _ = r.Forward(im)
			}
		})
		timeIt("beam_"+set, func() {
			for _, o := range outs {
				_, _ = r.BeamDecode(o)
			}
		})
		timeIt("recognize_"+set, func() {
			for _, im := range imgs {
				_, _ = r.Recognize(im)
			}
		})
		// Go has no API that reuses buffers between calls, so this is plain r.Recognize again: the
		// class exists so that the class join of bench.sh matches the Rust side's
		// recognize_reuse_*, which keeps one Workspace and Line across all calls.
		timeIt("recognize_reuse_"+set, func() {
			for _, im := range imgs {
				_, _ = r.Recognize(im)
			}
		})
	}
	raw, err := os.ReadFile(filepath.Join(testdata, "golden", "gray.bin"))
	if err != nil {
		fatal(err)
	}
	w, h := int(binary.LittleEndian.Uint32(raw)), int(binary.LittleEndian.Uint32(raw[4:]))
	g := imaging.NewBitmap(w, h, 8)
	copy(g.Pix, raw[12:12+w*h])
	bin := imaging.Otsu(g)
	eroded := imaging.Erode(bin, 3, 3)
	timeIt("otsu", func() { imaging.Otsu(g) })
	timeIt("sauvola_15", func() { imaging.Sauvola(g, 15, 0.34) })
	timeIt("skew_angle", func() { imaging.SkewAngle(bin) })
	timeIt("rotate", func() { imaging.Rotate(bin, 0.02) })
	timeIt("scale_0.5", func() { imaging.ScaleGray(g, 0.5) })
	timeIt("scale_1.7", func() { imaging.ScaleGray(g, 1.7) })
	timeIt("dilate_3x3", func() { imaging.Dilate(bin, 3, 3) })
	timeIt("erode_3x3", func() { imaging.Erode(bin, 3, 3) })
	timeIt("open_5x5", func() { imaging.Open(bin, 5, 5) })
	timeIt("close_5x5", func() { imaging.Close(bin, 5, 5) })
	timeIt("conn_comp_8", func() { imaging.ConnComp(bin, 8) })
	timeIt("distance", func() { imaging.DistanceFunction(bin) })
	timeIt("seed_fill_8", func() { imaging.SeedFill(eroded, bin, 8) })
}
