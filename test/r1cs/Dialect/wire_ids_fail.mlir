// RUN: r1cs-opt -split-input-file -verify-diagnostics %s -o /dev/null

// Every input must have an ID when explicit input IDs are supplied.
module {
  // expected-error@+1 {{'r1cs.input_labels' must contain one i64 label per input}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal, %b: !r1cs.signal) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: 2>}
}

// -----

// Explicit input IDs must be unique.
module {
  // expected-error@+1 {{'r1cs.input_labels' must not contain duplicate labels}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal, %b: !r1cs.signal) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: 2, 2>}
}

// -----

// Wire zero is reserved for the constant one.
module {
  // expected-error@+1 {{'r1cs.input_labels' must be in [1, 4294967295]}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: 0>}
}

// -----

// Input IDs cannot be negative.
module {
  // expected-error@+1 {{'r1cs.input_labels' must be in [1, 4294967295]}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: -1>}
}

// -----

// Input IDs must fit in uint32.
module {
  // expected-error@+1 {{'r1cs.input_labels' must be in [1, 4294967295]}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: 4294967296>}
}

// -----

// Public outputs precede inputs.
module {
  // expected-error@+1 {{wire IDs must follow public outputs, public inputs, private inputs, then internal wires}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal) {
    %out = r1cs.def 2 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: 1>}
}

// -----

// Public input IDs precede private input IDs even if their arguments are interleaved.
module {
  // expected-error@+1 {{wire IDs must follow public outputs, public inputs, private inputs, then internal wires}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal, %b: !r1cs.signal {pub = #r1cs.pub}) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
  } {r1cs.input_labels = array<i64: 2, 3>}
}

// -----

// Explicit wire IDs cannot leave gaps before internal wires.
module {
  // expected-error@+1 {{wire IDs must follow public outputs, public inputs, private inputs, then internal wires}}
  r1cs.circuit @Main inputs (%a: !r1cs.signal) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %internal = r1cs.def 4 : !r1cs.signal
  } {r1cs.input_labels = array<i64: 2>}
}
