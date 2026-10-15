#include <cstdint>
#include <tuple>

enum class Thing : std::uint8_t { One, Two, Three };

struct A {
  Thing th;
};

template <std::pair<Thing, Thing>... Pairs> static bool areOneOf(const A &a, const A &b) {
  return ((a.th == std::get<0>(Pairs) && b.th == std::get<1>(Pairs)) || ...);
}

int main() {
  A a, b;

  (void)areOneOf<{Thing::One, Thing::Two}, {Thing::One, Thing::Three}>(a, b);
}
