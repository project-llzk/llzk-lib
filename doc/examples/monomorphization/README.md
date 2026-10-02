# Arrays of differently specialized structs

These LLZK examples describe array fields whose element parameters depend on the
array indices. Each position selects a different concrete definition. They exercise
definition specialization directly, without depending on the Circom frontend.

They are schema and discovery examples: the constructors allocate struct handles
and demonstrate rolled family calls, but do not initialize the fields or implement
witness computation or constraints.

## One-dimensional bank

[widths.llzk](widths.llzk) describes this conceptual structure:

```text
Lane<Width> {
  values: felt[Width]
}
Bank {
  lanes[i]: Lane<2*i + 1>,  0 <= i < 4
}
```

The actual field declaration is:

```mlir
#width = affine_map<(i) -> (2 * i + 1)>
struct.member @lanes : !array.type<4 x !struct.type<@Lane::@Lane<[#width]>>>
```

| Index | Concrete element definition | Element's `values` field |
| --- | --- | --- |
| 0 | `Lane<1>` | `felt[1]` |
| 1 | `Lane<3>` | `felt[3]` |
| 2 | `Lane<5>` | `felt[5]` |
| 3 | `Lane<7>` | `felt[7]` |

The pass creates five specialized definitions: the bank and four lanes. The bank's
array field retains its affine element type and gains `poly.family` metadata mapping
indices `[0]` through `[3]` to the lane specialization IDs. Its single constructor
loop stays rolled; the call gains the four candidate IDs in
`poly.family_specializations`.

## Two-dimensional board with two template parameters

[grid.llzk](grid.llzk) describes:

```text
Tile<Rows, Cols> {
  cells: felt[Rows][Cols]
}
Board {
  tiles[i,j]: Tile<i + 1, j + 2>,  0 <= i < 2, 0 <= j < 3
}
```

| Array position | Concrete element definition | Element's `cells` field |
| --- | --- | --- |
| `[0,0]` | `Tile<1,2>` | `felt[1][2]` |
| `[0,1]` | `Tile<1,3>` | `felt[1][3]` |
| `[0,2]` | `Tile<1,4>` | `felt[1][4]` |
| `[1,0]` | `Tile<2,2>` | `felt[2][2]` |
| `[1,1]` | `Tile<2,3>` | `felt[2][3]` |
| `[1,2]` | `Tile<2,4>` | `felt[2][4]` |

The pass creates seven specialized definitions: the board and six tiles. Both
constructor loops remain rolled. The field's `poly.family` records the six index
pairs, and the constructor call records the same six candidate specialization IDs.

## Running the examples

From the repository root, with the built `result/bin/llzk-opt`:

```sh
nix develop .#release --command bash -c \
  'result/bin/llzk-opt doc/examples/monomorphization/widths.llzk --llzk-monomorphize=report=true -o /tmp/widths-specialized.llzk'
nix develop .#release --command bash -c \
  'result/bin/llzk-opt doc/examples/monomorphization/grid.llzk --llzk-monomorphize=report=true -o /tmp/grid-specialized.llzk'
```

Both examples were translated and verified with the Release tool. The reported
specialization counts were five and seven respectively. Original templates remain
in the output as well. IDs are module-local identities, not globally stable names.

The output describes which definition belongs at each array position. It does not
unroll the arrays into separate fields or choose a heterogeneous runtime layout;
those are later instance-elaboration concerns.
