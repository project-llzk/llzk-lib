{ stdenv, lib, cmake, ninja, mlir_pkg, llzk_pkg }:

let
  buildTypeStr = lib.toLower (mlir_pkg.cmakeBuildType or "release");
in
stdenv.mkDerivation {
  pname = "llzk-installcheck-${buildTypeStr}";
  version = "1.0.0";

  src = lib.cleanSource ./.;

  buildInputs = [ mlir_pkg llzk_pkg ];
  nativeBuildInputs = [ cmake ninja ];

  installPhase = ''touch "$out"'';
}
