# WTNS witness output

`llzk-witgen --output-wtns <file>` writes the version 2 binary witness format
consumed by snarkjs. The implementation follows the
[snarkjs WTNS writer](https://github.com/iden3/snarkjs/blob/master/src/wtns_utils.js).

## File schema

All integers and field elements are little-endian.

| Field | Encoding |
|---|---|
| Magic | Four bytes: `wtns` |
| Version | `u32`, currently 2 |
| Section count | `u32`, currently 2 |

Each section begins with a `u32` section type and a `u64` payload size.

Section 1 is the witness header:

| Field | Encoding |
|---|---|
| Field element width | `u32` bytes, rounded up to a 64-bit limb |
| Prime modulus | Exactly the field element width in bytes |
| Witness length | `u32` |

Section 2 contains `witness length` field elements, each encoded using the
width declared by section 1.

## Wire ordering

The witness order must match LLZK's R1CS binary exporter:

1. Implicit constant-one wire.
2. Public output members, in main-struct declaration order.
3. Public inputs, in main-function argument order.
4. Private inputs, in main-function argument order.
5. Non-public output members, in main-struct declaration order.

Input visibility comes from the corresponding arguments of `@constrain`, as it
does in the R1CS lowering pass. The first `@constrain` argument is `self` and is
not an R1CS input.

For evaluated modules (`poly.evaluated_main`), lowering produces a
`poly.wire_bindings` map from physical wires to input or nested component
storage paths. The interpreter materializes degree-lowering and R1CS auxiliary
members and the WTNS writer follows that map, including array and POD leaves.
Input-rooted auxiliary reads capture entry values before compute mutates local
storage; function calls preserve aggregate argument value semantics.

The legacy, non-evaluated path supports scalar felt inputs and main members.
It cross-checks witness length against an R1CS-lowered module clone and rejects
aggregates or auxiliary wires it cannot materialize. The evaluated path checks
binding order and diagnoses missing storage values before serialization.

Evaluation may grant `llzk.pub` access to nested members so generated straightline
constraints can read them. Circuit visibility remains in `poly.original_public`;
public witness serialization uses that original visibility. `poly.evaluated_main`
is the authoritative module-state marker; function markers record provenance.

The interpreter uses copy-on-write aggregate storage. Value copies share contents
until mutation, while SSA aliases refer to the same aggregate wrapper.
