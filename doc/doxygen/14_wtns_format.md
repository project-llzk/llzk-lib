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

The current WTNS writer supports scalar felt inputs and main members. It also
cross-checks its witness length against an R1CS-lowered clone of the module.
Circuits that require synthesized R1CS auxiliary wires are rejected until
witgen can materialize those auxiliary values. Unsupported aggregates and wire
count mismatches are therefore diagnosed rather than silently serialized.
