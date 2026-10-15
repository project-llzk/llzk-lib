{
  lib,
  runCommand,
  python3,
  mlir_pkg,
  llzk_pkg,
}:

# Run the same assertions against installed packages, outside either build tree.
runCommand "llzk-python-installcheck"
  {
    nativeBuildInputs = [ python3 ];
    PYTHONPATH = "${llzk_pkg}/python:" + lib.makeSearchPath python3.sitePackages (
      [ mlir_pkg.python ] ++ mlir_pkg.pythonDeps
    );
  }
  ''
    python ${../../test/Python/registration.py}
    python ${../../test/Python/operations.py}
    touch "$out"
  ''
