//! Buffers that a call keeps for the next one, so that a call on a warm [`crate::recog::Workspace`]
//! allocates nothing.
//!
//! Every buffer that grows with the input is charged to the call's [`Budget`] when its capacity
//! grows, never when it is reused; the capacity a workspace already holds is charged once, at the
//! start of the call (see `Recognizer::recognize_into`).

use crate::{Budget, Error};

/// Empties `v` and makes sure it has room for `n` elements, charging the growth in capacity. A vec
/// that has to grow is freed first, so the old block and the new one are never live together and
/// the charge (the new capacity less the old one, which the caller has charged already) covers
/// the heap.
pub(crate) fn grow<T>(budget: &Budget, v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    v.clear();
    if v.capacity() >= n {
        return Ok(());
    }
    let size = std::mem::size_of::<T>();
    let old = v.capacity() * size;
    let delta = n.checked_mul(size).ok_or(Error::AllocLimit)? - old;
    budget.charge(delta)?;
    *v = Vec::new();
    if v.try_reserve_exact(n).is_err() {
        budget.release(delta);
        return Err(Error::OutOfMemory);
    }
    Ok(())
}

/// `v` as `n` copies of `x`, in the capacity it has or charged growth (see [`grow`]).
pub(crate) fn refill<T: Clone>(
    budget: &Budget,
    v: &mut Vec<T>,
    n: usize,
    x: T,
) -> Result<(), Error> {
    grow(budget, v, n)?;
    v.resize(n, x);
    Ok(())
}

/// Keeps a vec that a call has nearly filled from being grown by the next one: past three
/// quarters full its capacity doubles, so that a later call whose output is a third larger than
/// this one's (the network's random border taps move the beam a little from call to call) still
/// fits. Best effort: a call never fails for want of it, and the growth is charged like any other.
pub(crate) fn headroom<T>(budget: &Budget, v: &mut Vec<T>) {
    if v.len().saturating_mul(4) > v.capacity().saturating_mul(3) {
        let _ = budget.reserve(v, v.capacity().max(4));
    }
}

/// Appends `src` to `v`, charging the growth when it does not fit in the capacity `v` has.
pub(crate) fn extend_charged<T: Copy>(
    budget: &Budget,
    v: &mut Vec<T>,
    src: &[T],
) -> Result<(), Error> {
    if v.capacity() - v.len() < src.len() {
        budget.reserve(v, src.len())?;
    }
    v.extend_from_slice(src);
    Ok(())
}

/// Sets `s` to `t`, charging the growth of its capacity. A string that has to grow is freed
/// first, as in [`grow`].
pub(crate) fn set_str(budget: &Budget, s: &mut String, t: &str) -> Result<(), Error> {
    s.clear();
    if s.capacity() < t.len() {
        let delta = t.len() - s.capacity();
        budget.charge(delta)?;
        *s = String::new();
        if s.try_reserve_exact(t.len()).is_err() {
            budget.release(delta);
            return Err(Error::OutOfMemory);
        }
    }
    s.push_str(t);
    Ok(())
}

/// Sets `s` to the concatenation of `texts`, as [`set_str`].
pub(crate) fn set_joined<'a>(
    budget: &Budget,
    s: &mut String,
    texts: impl Iterator<Item = &'a str> + Clone,
) -> Result<(), Error> {
    let len: usize = texts.clone().map(str::len).sum();
    s.clear();
    if s.capacity() < len {
        let delta = len - s.capacity();
        budget.charge(delta)?;
        *s = String::new();
        if s.try_reserve_exact(len).is_err() {
            budget.release(delta);
            return Err(Error::OutOfMemory);
        }
    }
    for t in texts {
        s.push_str(t);
    }
    Ok(())
}

#[derive(Debug, Default)]
struct Slot<T> {
    buf: Vec<T>,
    /// The address of the buffer while it is handed out, 0 while it is in the slot.
    taken: usize,
}

/// Buffers handed out in the order a call asks for them: a request takes the lowest slot that is
/// free and a buffer goes back to the slot it came from. Which slot serves which request depends
/// only on the sequence of takes and gives, never on the sizes, so a call that repeats the sequence
/// of an earlier one finds every slot as large as its requests need and allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct SlotPool<T> {
    slots: Vec<Slot<T>>,
}

impl<T: Clone> SlotPool<T> {
    /// Marks every slot free, as at the start of a call. A buffer that was not given back (the
    /// call ended in an error) stays lost: its slot is empty and grows again.
    pub(crate) fn reset(&mut self) {
        for s in &mut self.slots {
            s.taken = 0;
        }
    }

    /// The heap the slots hold, by capacity.
    pub(crate) fn bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.buf.capacity() * std::mem::size_of::<T>())
            .sum::<usize>()
            + self.slots.capacity() * std::mem::size_of::<Slot<T>>()
    }

    /// `n` copies of `fill` in the lowest free slot, charging the growth of its capacity.
    pub(crate) fn take(&mut self, budget: &Budget, n: usize, fill: T) -> Result<Vec<T>, Error> {
        let j = match self.slots.iter().position(|s| s.taken == 0) {
            Some(j) => j,
            None => {
                budget.push(
                    &mut self.slots,
                    Slot {
                        buf: Vec::new(),
                        taken: 0,
                    },
                )?;
                self.slots.len() - 1
            }
        };
        let slot = &mut self.slots[j];
        let mut v = std::mem::take(&mut slot.buf);
        grow(budget, &mut v, n)?;
        v.resize(n, fill);
        slot.taken = v.as_ptr() as usize;
        Ok(v)
    }

    /// Puts `v` back in the slot it was taken from. A vec that is not one of the slots' (it grew
    /// and moved) is dropped.
    pub(crate) fn give(&mut self, v: Vec<T>) {
        let p = v.as_ptr() as usize;
        if let Some(s) = self.slots.iter_mut().find(|s| s.taken == p && p != 0) {
            s.buf = v;
            s.taken = 0;
        }
    }
}
