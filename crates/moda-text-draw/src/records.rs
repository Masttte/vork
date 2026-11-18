//! Retained GPU record columns shared by text and the engine packet writer.

/// One storage buffer of a packet as the CPU holds it: the records its collect uploaded, kept so a patch compares
/// a changed item's record with the one the buffer holds and writes it where it stands. Record `i` is the
/// buffer's bytes from `i * size_of::<T>()`.
pub struct Column<T> {
    records: Vec<T>,
    /// Records a patch asked for and has not committed, by index (`Column::put`).
    pending: Vec<(u32, T)>,
}

impl<T> Default for Column<T> {
    fn default() -> Self {
        Self { records: Vec::new(), pending: Vec::new() }
    }
}

impl<T: bytemuck::Pod> Column<T> {
    pub fn of(records: Vec<T>) -> Self {
        Self { records, pending: Vec::new() }
    }

    /// Ask for `record` at `index`. `Ok(true)`: it differs from the one held and is pending until
    /// [`Column::commit`]; `Ok(false)`: the buffer already holds it. `Err`: the column has no such record.
    pub fn put(&mut self, index: u32, record: &T) -> std::result::Result<bool, &'static str> {
        let held = self.records.get(index as usize).ok_or("record")?;
        if bytemuck::bytes_of(held) == bytemuck::bytes_of(record) {
            return Ok(false);
        }
        self.pending.push((index, *record));
        Ok(true)
    }
}

/// A [`Column`] without its record type, for what is done to every column of a packet alike.
pub trait PendingRecords {
    /// The bytes the column's buffer holds.
    fn bytes(&self) -> u64;
    fn is_pending(&self) -> bool;
    /// Forget what was asked for.
    fn discard(&mut self);
    /// Take the pending records into the column and hand `upload` the column from its first changed record to
    /// its last, as (byte offset in the buffer, bytes): one write for a column, however its changed records lie.
    fn commit(&mut self, upload: &mut dyn FnMut(u64, &[u8]));
}

impl<T: bytemuck::Pod> PendingRecords for Column<T> {
    fn bytes(&self) -> u64 {
        std::mem::size_of_val(self.records.as_slice()) as u64
    }
    fn is_pending(&self) -> bool {
        !self.pending.is_empty()
    }
    fn discard(&mut self) {
        self.pending.clear();
    }
    fn commit(&mut self, upload: &mut dyn FnMut(u64, &[u8])) {
        let mut span: Option<(usize, usize)> = None;
        for (index, record) in self.pending.drain(..) {
            let index = index as usize;
            self.records[index] = record;
            span = Some(span.map_or((index, index + 1), |(from, to)| (from.min(index), to.max(index + 1))));
        }
        if let Some((from, to)) = span {
            let size = std::mem::size_of::<T>() as u64;
            upload(from as u64 * size, bytemuck::cast_slice(&self.records[from..to]));
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_column_uploads_once_from_its_first_changed_record_to_its_last() {
        let mut column = Column::of((0..8u32).map(|i| [i, 0]).collect::<Vec<[u32; 2]>>());
        // A record the buffer already holds asks for nothing.
        assert_eq!(column.put(3, &[3, 0]), Ok(false));
        assert!(!column.is_pending());
        // Asked out of order, with unchanged records between them.
        assert_eq!(column.put(5, &[5, 9]), Ok(true));
        assert_eq!(column.put(2, &[2, 9]), Ok(true));
        assert_eq!(column.put(1, &[1, 9]), Ok(true));
        assert_eq!(column.put(8, &[8, 9]), Err("record"));
        // Nothing is taken before the commit.
        assert_eq!(column.records[2], [2, 0]);
        let mut uploads: Vec<(u64, Vec<u8>)> = Vec::new();
        column.commit(&mut |offset, data| uploads.push((offset, data.to_vec())));
        let bytes = |records: &[[u32; 2]]| bytemuck::cast_slice::<[u32; 2], u8>(records).to_vec();
        assert_eq!(uploads, vec![(8, bytes(&[[1, 9], [2, 9], [3, 0], [4, 0], [5, 9]]))]);
        assert!(!column.is_pending());
        assert_eq!(column.records, vec![[0, 0], [1, 9], [2, 9], [3, 0], [4, 0], [5, 9], [6, 0], [7, 0]]);
        assert_eq!(column.bytes(), 64);
        // A discarded record is never taken, and a column with nothing pending uploads nothing.
        assert_eq!(column.put(0, &[7, 7]), Ok(true));
        column.discard();
        column.commit(&mut |_, _| panic!("nothing is pending"));
        assert_eq!(column.records[0], [0, 0]);
    }

}
