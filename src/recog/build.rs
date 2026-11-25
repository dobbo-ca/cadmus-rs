//! A runnable network from a parsed `lstm` component: the `Network::CreateFromFile` dispatch of
//! `src/lstm/network.cpp` (Apache-2.0, see NOTICE).

use crate::imaging::check_cancel;
use crate::nn::{self, Activation, Layer, ReversalKind};
use crate::tessdata::{self, LayerType, short};
use crate::{Budget, Error, Limits};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn err(msg: impl Into<String>) -> Error {
    Error::Recog(msg.into())
}

/// A loaded, runnable recognizer graph plus the scalars the decoder needs out of the LSTM
/// component's trailer.
#[derive(Debug)]
pub struct Network {
    pub root: Layer,
    pub input_height: usize,
    pub num_outputs: usize,
    pub null_char: usize,
    pub x_scale: usize,
    /// The recognizer's `TRand`. `normalize` draws from it first (for rows the scaler left short)
    /// and every `Convolve` continues from there, exactly as `Copy2DImage` precedes
    /// `Convolve::Forward` in Tesseract. It is shared state: its draws advance with every call.
    pub rand: Arc<nn::Rand>,
}

impl Network {
    /// Converts a parsed layer tree into a runnable graph, charging the copied weights against
    /// `limits.max_alloc`. The cancel flag is checked at every layer.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` when the copy does not fit, `Error::Cancelled`, else `Error::Recog` for
    /// a graph the runtime cannot run, `Error::OutOfMemory` if the allocator refuses.
    pub fn build(
        rec: &tessdata::Recognizer,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Network, Error> {
        Network::build_in(rec, &Budget::new(limits), cancel)
    }

    /// `build`, charging the call's own `budget`.
    pub(crate) fn build_in(
        rec: &tessdata::Recognizer,
        budget: &Budget,
        cancel: &AtomicBool,
    ) -> Result<Network, Error> {
        if rec.training_flags & 1 != 0 {
            return Err(err(format!(
                "model has TF_INT_MODE set (training_flags={}); only float weights from tessdata_best are supported",
                rec.training_flags
            )));
        }
        // The Arc holds two counts and the generator.
        budget.charge(2 * std::mem::size_of::<usize>() + std::mem::size_of::<nn::Rand>())?;
        let rand = Arc::new(nn::Rand::new(
            (i64::from(rec.sample_iteration) * 0x10000001) as u64,
        ));
        budget.charge(std::mem::size_of::<Layer>())?;
        let root = build_layer(&rec.network, &rand, budget, cancel)?;

        let shape =
            find_input_shape(&rec.network).ok_or_else(|| err("model has no Input layer"))?;
        if shape.height <= 0 {
            return Err(err(format!(
                "input shape height is {}; a fixed height is required",
                shape.height
            )));
        }
        if shape.depth != 1 {
            return Err(err(format!(
                "input depth {}; only 1-channel grey input is supported",
                shape.depth
            )));
        }
        // A layer whose size does not fit saturates to usize::MAX (a Reconfig or Convolve with
        // huge scales, alone or summed by a Parallel); no network has that many outputs.
        let num_outputs = root.num_outputs();
        if num_outputs == usize::MAX {
            return Err(err("the network's output count overflows"));
        }
        let n = Network {
            input_height: shape.height as usize,
            num_outputs,
            null_char: rec.null_char as usize,
            x_scale: x_scale_factor(&rec.network),
            root,
            rand,
        };
        // null_char_ is the authority for the CTC blank; it is structurally free to differ from
        // num_outputs-1, so it is read rather than assumed. It must be a valid output index.
        if rec.null_char < 0 || n.null_char >= n.num_outputs {
            return Err(err(format!(
                "null_char {} is outside the {} network outputs",
                rec.null_char, n.num_outputs
            )));
        }
        Ok(n)
    }
}

/// Builds `l` and everything under it. The layer's own slot (in its parent's stack, or the
/// root's) is charged by whoever holds it; its name, its box and its weights are charged here.
fn build_layer(
    l: &tessdata::Layer,
    rnd: &Arc<nn::Rand>,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Layer, Error> {
    use LayerType as T;
    check_cancel(cancel)?;
    // Every layer keeps its own copy of the name.
    budget.charge(l.name.len())?;
    match l.typ {
        T::Input => Ok(Layer::Input(nn::Input::new(&l.name, l.num_outputs))),

        T::Series | T::Parallel | T::Replicated => {
            let stack = build_stack(l, rnd, budget, cancel)?;
            if l.typ == T::Series {
                Ok(Layer::Series(nn::Series::new(&l.name, stack)?))
            } else {
                Ok(Layer::Parallel(nn::Parallel::new(&l.name, stack)?))
            }
        }

        T::XReversed | T::YReversed | T::XyTranspose => {
            let mut stack = build_stack(l, rnd, budget, cancel)?;
            if stack.len() != 1 {
                return Err(err(format!(
                    "{} {} has {} children; want exactly 1",
                    l.typ,
                    short(&l.name),
                    stack.len()
                )));
            }
            let kind = match l.typ {
                T::XReversed => ReversalKind::ReverseX,
                T::YReversed => ReversalKind::ReverseY,
                _ => ReversalKind::TransposeXY,
            };
            let sub = stack.remove(0);
            // The sub-layer goes into a box.
            budget.charge(std::mem::size_of::<Layer>())?;
            Ok(Layer::Reversed(nn::Reversed::new(&l.name, kind, sub)))
        }

        T::Convolve => Ok(Layer::Convolve(nn::Convolve::new(
            &l.name,
            l.num_inputs,
            non_negative(l.half_x, &l.name)?,
            non_negative(l.half_y, &l.name)?,
            Arc::clone(rnd),
        ))),

        T::Maxpool => Ok(Layer::Reconfig(nn::Reconfig::new_maxpool(
            &l.name,
            l.num_inputs,
            non_negative(l.x_scale, &l.name)?,
            non_negative(l.y_scale, &l.name)?,
        ))),

        T::Reconfig => Ok(Layer::Reconfig(nn::Reconfig::new(
            &l.name,
            l.num_inputs,
            non_negative(l.x_scale, &l.name)?,
            non_negative(l.y_scale, &l.name)?,
        ))),

        T::Softmax
        | T::SoftmaxNoCtc
        | T::Tanh
        | T::Relu
        | T::Linear
        | T::Logistic
        | T::PosClip
        | T::SymClip => {
            if l.matrices.len() != 1 {
                return Err(err(format!(
                    "{} {} has {} weight matrices; want 1",
                    l.typ,
                    short(&l.name),
                    l.matrices.len()
                )));
            }
            let m = convert_matrix(&l.name, &l.matrices[0], budget)?;
            // FullyConnected's matrix is no x (ni+1). Both dimensions are checked because a
            // mismatch means the graph parse or the weight load slipped.
            if m.outputs != l.num_outputs || m.inputs != l.num_inputs {
                return Err(err(format!(
                    "{} {} matrix is {}x{}; want {}x{} from the layer header",
                    l.typ,
                    short(&l.name),
                    m.outputs,
                    m.inputs + 1,
                    l.num_outputs,
                    l.num_inputs + 1
                )));
            }
            let act = match l.typ {
                T::Softmax | T::SoftmaxNoCtc => Activation::Softmax,
                T::Tanh => Activation::Tanh,
                T::Relu => Activation::Relu,
                T::Linear => Activation::Linear,
                T::Logistic => Activation::Logistic,
                T::PosClip => Activation::PosClip,
                _ => Activation::SymClip,
            };
            check_num_weights(l)?;
            Ok(Layer::FullyConnected(nn::FullyConnected::new(
                &l.name, act, m,
            )))
        }

        T::Lstm | T::LstmSummary => {
            if l.matrices.len() != 4 {
                return Err(err(format!(
                    "LSTM {} has {} gate matrices; want 4 (a 5th means a 2-D layer, which is out of scope)",
                    short(&l.name),
                    l.matrices.len()
                )));
            }
            let mut gates = Vec::new();
            budget.reserve(&mut gates, 4)?;
            for (i, src) in l.matrices.iter().enumerate() {
                let m = convert_matrix(&l.name, src, budget)?;
                if m.inputs as i64 != i64::from(l.na) {
                    return Err(err(format!(
                        "LSTM {} gate {i} has {} input columns; want na={}",
                        short(&l.name),
                        m.inputs,
                        l.na
                    )));
                }
                gates.push(m);
            }
            check_num_weights(l)?;
            let gates: [nn::Matrix; 4] = gates.try_into().map_err(|_| err("LSTM gate count"))?;
            Ok(Layer::Lstm(nn::Lstm::new(
                &l.name,
                l.num_inputs,
                l.na as usize,
                l.typ == T::LstmSummary,
                gates,
            )?))
        }

        _ => Err(err(format!(
            "layer type {} ({}) is not supported by the runtime",
            l.typ,
            short(&l.name)
        ))),
    }
}

fn non_negative(v: i32, name: &str) -> Result<usize, Error> {
    usize::try_from(v).map_err(|_| err(format!("layer {} has a negative size {v}", short(name))))
}

fn build_stack(
    l: &tessdata::Layer,
    rnd: &Arc<nn::Rand>,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Vec<Layer>, Error> {
    // The room for the children's slots is charged once, here.
    let mut stack = Vec::new();
    budget.reserve(&mut stack, l.children.len())?;
    for (i, c) in l.children.iter().enumerate() {
        let sub = build_layer(c, rnd, budget, cancel).map_err(|e| match e {
            Error::Recog(s) => err(format!("{} {} child {i}: {s}", l.typ, short(&l.name))),
            e => e,
        })?;
        stack.push(sub);
    }
    Ok(stack)
}

/// A layer's serialized `num_weights` against the element count of the matrices actually loaded,
/// bias columns included. Free validation that the weight load stayed aligned.
fn check_num_weights(l: &tessdata::Layer) -> Result<(), Error> {
    let total: usize = l.matrices.iter().map(|m| m.rows * m.cols).sum();
    if total != l.num_weights {
        return Err(err(format!(
            "layer {} matrices hold {total} weights but the header says {}",
            short(&l.name),
            l.num_weights
        )));
    }
    Ok(())
}

/// Reshapes a loader matrix into a runtime one. The loader's `cols` counts the bias column; the
/// runtime's `inputs` does not.
fn convert_matrix(name: &str, m: &tessdata::Matrix, budget: &Budget) -> Result<nn::Matrix, Error> {
    if m.cols < 1 {
        return Err(err(format!(
            "layer {} has a matrix with {} columns",
            short(name),
            m.cols
        )));
    }
    let mut w = budget.vec(m.values.len(), 0f64)?;
    w.copy_from_slice(&m.values);
    nn::Matrix::new(m.rows, m.cols - 1, w, budget.limits())
}

fn find_input_shape(l: &tessdata::Layer) -> Option<tessdata::InputShape> {
    if let Some(s) = l.shape {
        return Some(s);
    }
    l.children.iter().find_map(find_input_shape)
}

/// `Network::XScaleFactor`, and the Series case is the ONLY one that multiplies: `Reconfig` (and
/// Maxpool) give `x_scale_`, `Plumbing` the first child's, `Series` the product over the stack.
/// Parallel, Replicated and the three Reversed variants take the FIRST child only.
fn x_scale_factor(l: &tessdata::Layer) -> usize {
    match l.typ {
        LayerType::Maxpool | LayerType::Reconfig => l.x_scale.max(0) as usize,
        LayerType::Series => l
            .children
            .iter()
            .map(x_scale_factor)
            .fold(1usize, usize::saturating_mul),
        LayerType::Parallel
        | LayerType::Replicated
        | LayerType::ParRlLstm
        | LayerType::ParUdLstm
        | LayerType::Par2dLstm
        | LayerType::XReversed
        | LayerType::YReversed
        | LayerType::XyTranspose => l.children.first().map_or(1, x_scale_factor),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_weight_is_accepted_when_the_network_is_built() {
        let m = |v: f64| tessdata::Matrix {
            rows: 1,
            cols: 2,
            values: vec![1.0, v],
        };
        let budget = Budget::new(&Limits::default());
        assert!(convert_matrix("f", &m(0.5), &budget).is_ok());
        // Go builds it too; the forward pass fails only if the NaN reaches a tabulated activation.
        assert!(convert_matrix("f", &m(f64::NAN), &budget).is_ok());
    }

    /// A serialized layer: the type tag, the training and backprop bytes, the flags, input,
    /// output and weight counts and the name (`Network::Serialize`), then the layer's body.
    fn layer(typ: LayerType, ni: i32, no: i32, name: &str, body: &[u8]) -> Vec<u8> {
        let mut b = vec![typ as u8, 0, 0];
        for v in [0, ni, no, 0, name.len() as i32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(body);
        b
    }

    fn ints(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A Parallel root over an Input and a Reconfig whose sizes saturate. Summing the children's
    /// output counts overflowed (a panic with overflow checks on, as in tests and cargo-fuzz).
    #[test]
    fn a_parallel_with_a_saturating_child_is_rejected_when_the_network_is_built() {
        let input = layer(LayerType::Input, 0, 1, "in", &ints(&[1, 36, 0, 1, 0]));
        let reconfig = layer(
            LayerType::Reconfig,
            i32::MAX,
            0,
            "rc",
            &ints(&[i32::MAX, i32::MAX]),
        );
        let mut body = 2u32.to_le_bytes().to_vec();
        body.extend(input);
        body.extend(reconfig);
        let mut data = layer(LayerType::Parallel, 1, 1, "par", &body);
        // The trailer: the spec string, TF_COMPRESS_UNICHARSET, two counters, null_char 0, and
        // the three f32 fields.
        data.extend(3u32.to_le_bytes());
        data.extend(b"[x]");
        data.extend(ints(&[64, 0, 0, 0]));
        data.extend([0u8; 12]);

        let (limits, cancel) = (Limits::default(), AtomicBool::new(false));
        let budget = Budget::new(&limits);
        let rec = tessdata::parse_recognizer(&data, false, &budget, &cancel).unwrap();
        let e = Network::build(&rec, &limits, &cancel).unwrap_err();
        assert!(
            matches!(&e, Error::Recog(m) if m.contains("overflows")),
            "{e}"
        );
    }
}
