# padding_ratio — Design Note

## What it does

`padding_ratio` controls how much extra empty space each row gets
beyond its current live entries.

**Formula:** `total_slots = ceil(nnz × (1 + padding_ratio))`

Example: a row with 10 live entries and `padding_ratio=0.2` gets
`ceil(10 × 1.2) = 12` slots — 10 live + 2 empty padding slots.

## When to increase it

- You are running aggressive Dynamic Sparse Training (DST) with
  high connection churn (many grow/drop operations per step)
- Rows hit capacity frequently, triggering expensive `rewrite_row` calls
- If `rewrite_row` is being called often during DST updates, increase `padding_ratio` to reduce rewrite frequency.

## When to decrease it

- Memory is constrained (each extra slot costs 4 bytes for float32
  values + 4 bytes for col_indices)
- Your sparsity pattern is static or low-churn
- You are doing inference only (no mutation needed)

## Default: 0.2

20% padding empirically balances memory overhead against grow frequency
for typical DST schedules with ~10% connection churn per step.

## Memory cost

For a matrix with `nnz` live entries:

| padding_ratio | total_slots | overhead |
|--------------|-------------|----------|
| 0.0 | nnz | 0% |
| 0.1 | ceil(nnz × 1.1) | 10% |
| 0.2 | ceil(nnz × 1.2) | 20% |
| 0.5 | ceil(nnz × 1.5) | 50% |

Memory in bytes: `total_slots × 8` (4 bytes values + 4 bytes col_indices)
