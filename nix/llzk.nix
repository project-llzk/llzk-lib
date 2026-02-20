{
  stdenv, lib,

  # build dependencies
  clang, cmake, ninja,
  mlir_pkg, nlohmann_json, pcl_pkg,

  # test dependencies
  gtest, python3, lit, z3, cvc5,

  cmakeBuildType ? "Release"
}:

stdenv.mkDerivation {
  pname = "llzk-${lib.toLower cmakeBuildType}";
  version = "0.1.0";
  src =
    let
      src0 = lib.cleanSource (builtins.path {
        path = ./..;
        name = "llzk-source";
      });
    in
      lib.cleanSourceWith {
        # Ignore unnecessary files
        filter = path: type: !(lib.lists.any (x: x) [
          (path == toString (src0.origSrc + "/README.md"))
          (type == "directory" && path == toString (src0.origSrc + "/third-party"))
          (type == "directory" && path == toString (src0.origSrc + "/.github"))
          (type == "regular" && lib.strings.hasSuffix ".nix" (toString path))
          (type == "regular" && baseNameOf path == "flake.lock")
        ]);
        src = src0;
      };

  nativeBuildInputs = [ cmake ninja ];
  buildInputs = [
    clang.dev mlir_pkg z3.lib pcl_pkg
  ] ++ lib.optionals mlir_pkg.hasPythonBindings [
    mlir_pkg.python
    mlir_pkg.pythonDeps
  ];

  propagatedBuildInputs = [ pcl_pkg ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=${cmakeBuildType}"
    "-DLLZK_BUILD_DEVTOOLS=ON"
  ];

  # Needed for mlir-tblgen to run properly.
  preBuild = ''
    export LD_LIBRARY_PATH=${z3.lib}/lib:$LD_LIBRARY_PATH
  '';

  # This is done specifically so that the configure phase can find /usr/bin/sw_vers,
  # which is MacOS specific.
  # Note that it's important for "/usr/bin/" to be last in the list so we don't
  # accidentally use the system clang, etc.
  preConfigure = ''
    if [[ "$(uname)" == "Darwin" ]]; then
      export OLD_PATH=$PATH
      export PATH="$PATH:/usr/bin/"
    fi
  '';

  # this undoes the above configuration, as it will cause problems later.
  postConfigure = ''
    if [[ "$(uname)" == "Darwin" ]]; then
      export PATH=$OLD_PATH
      # unset OLD_PATH
    fi
  '';

  doCheck = true;
  checkTarget = "check";
  checkInputs = [ gtest python3 lit z3 cvc5 ];
}
