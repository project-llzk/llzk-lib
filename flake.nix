{
  inputs = {
    llzk-pkgs.url = "github:project-llzk/llzk-nix-pkgs";
    nixpkgs.follows = "llzk-pkgs/nixpkgs";
    flake-utils.follows = "llzk-pkgs/flake-utils";

    release-helpers = {
      url = "github:project-llzk/open-source-release-helpers";
      inputs = {
        nixpkgs.follows = "llzk-pkgs/nixpkgs";
        flake-utils.follows = "llzk-pkgs/flake-utils";
      };
    };

    pcl-mlir-pkg = {
      url = "github:Veridise/pcl-mlir";
      inputs = {
        shared-pkgs.follows = "llzk-pkgs";
        nixpkgs.follows = "llzk-pkgs/nixpkgs";
        flake-utils.follows = "llzk-pkgs/flake-utils";
        release-helpers.follows = "release-helpers";
      };
    };
  };

  # Custom colored bash prompt
  nixConfig.bash-prompt = ''\[\e[0;32m\][LLZK]\[\e[m\] \[\e[38;5;244m\]\w\[\e[m\] % '';

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      llzk-pkgs,
      pcl-mlir-pkg,
      release-helpers,
    }:
    {
      # First, we define the packages used in this repository/flake
      overlays.llzk =
        final: prev:
        let
          mkLlzkDebWithSans =
            stdenv: reportName:
            (final.llzk-debug.override { inherit stdenv; }).overrideAttrs (attrs: {
              cmakeBuildType = "DebWithSans";

              # Disable container overflow checks because it can give false positives in
              # newGeneralRewritePatternSet() since LLVM itself is not built with ASan.
              # https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
              preBuild = ''
                export ASAN_OPTIONS=detect_container_overflow=0
              ''
              + attrs.preBuild;

              postInstall = ''
                if [ -f test/report.xml ]; then
                  mkdir -p $out/artifacts
                  echo "-- Copying xUnit report to $out/artifacts/${reportName}-report.xml"
                  cp test/report.xml $out/artifacts/${reportName}-report.xml
                fi
              '';
            });
        in
        {
          llzk = final.callPackage ./nix/llzk.nix {
            clang = final.llzk-llvmPackages.clang-unwrapped;
            mlir_pkg = final.mlir;
            pcl_pkg = final.pcl-mlir;
          };
          llzk-debug =
            (final.callPackage ./nix/llzk.nix {
              clang = final.llzk-llvmPackages-debug.clang-unwrapped;
              mlir_pkg = final.mlir-debug;
              pcl_pkg = final.pcl-mlir-debug;
              cmakeBuildType = "Debug";
            }).overrideAttrs
              (attrs: {
                NIX_CFLAGS_COMPILE = (attrs.NIX_CFLAGS_COMPILE or "") + " -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0";
              });

          mlir-with-python = final.mlir.override {
            enablePythonBindings = true;
          };
          llzk-with-python = final.llzk.override {
            mlir_pkg = final.mlir-with-python;
          };

          llzkDebWithSansGCC = mkLlzkDebWithSans final.gccStdenv "gcc";
          llzkDebWithSansClang = mkLlzkDebWithSans final.clangStdenv "clang";

          llzkDebWithSansClangCov = final.llzkDebWithSansClang.overrideAttrs (attrs: {
            postCheck = ''
              MANIFEST=profiles.manifest
              PROFDATA=coverage.profdata
              BINS=bins.lst
              if [[ "$(uname)" == "Darwin" ]]; then
                find bin lib -type f | xargs file | fgrep Mach-O | grep executable | cut -f1 -d: > $BINS
              else
                find bin lib -type f | xargs file | grep ELF | grep executable | cut -f1 -d: > $BINS
              fi
              echo -n "Found profraw files:"
              find test -name "*.profraw" | tee $MANIFEST | wc -l
              cat $MANIFEST
              llvm-profdata merge -sparse -f $MANIFEST -o $PROFDATA
              OBJS=$( (head -n 1 $BINS ; tail -n +2 $BINS | sed -e "s/^/-object /") | xargs)
              # TODO HTML reports
              llvm-cov report $OBJS -instr-profile $PROFDATA > cov-summary.txt
              echo =========== COVERAGE SUMMARY =================
              cat cov-summary.txt
              echo ==============================================
              llvm-cov export -format=lcov -instr-profile $PROFDATA $OBJS > report.lcov
              rm -rf $MANIFEST $PROFDATA $BINS
            '';

            postInstall = ''
              mkdir -p $out/artifacts/
              echo "-- Copying coverage summary to $out/artifacts/cov-summary.txt"
              cp cov-summary.txt $out/artifacts/
              echo "-- Copying lcov report to $out/artifacts/report.lcov"
              cp report.lcov $out/artifacts/
              if [ -f test/report.xml ]; then
                echo "-- Copying xUnit report to $out/artifacts/clang-report.xml"
                cp test/report.xml $out/artifacts/clang-report.xml
              fi
            '';
          });

          llzkDocs = final.llzk.overrideAttrs (attrs: {
            nativeBuildInputs = attrs.nativeBuildInputs ++ [
              final.doxygen
              final.graphviz
              final.git
              final.cacert
            ];
            buildPhase = ''
              cmake --build . --target doc -j$NIX_BUILD_CORES
            '';
            installPhase = ''
              cp -r ./doc $out
            '';
            doCheck = false;
          });

          ccacheStdenv = prev.ccacheStdenv.override {
            extraConfig = ''
              export CCACHE_DIR=/tmp/ccache
              export CCACHE_UMASK=007
              export CCACHE_COMPRESS=1
            '';
          };

          # The default shell is used for LLZK development.
          # Because `nix develop` is used to set up a dev shell for a given
          # derivation, we just need to extend the llzk derivation with any
          # extra tools we need.
          devShellBase = pkgs: llzkEnv: {
            shell = llzkEnv.overrideAttrs (old: {
              nativeBuildInputs =
                old.nativeBuildInputs
                ++ (with pkgs; [
                  doxygen
                  graphviz
                  git

                  # clang-tidy and clang-format
                  llzk-llvmPackages.clang-tools

                  # git-clang-format
                  libclang.python
                ]);

              shellHook = ''
                # needed to get accurate compile_commands.json
                export CXXFLAGS="$NIX_CFLAGS_COMPILE"

                # Add binary dir to PATH for convenience
                export PATH="$PWD"/build/bin:"$PATH"

                # Add release helpers to the PATH for convenience
                export PATH="${pkgs.changelogCreator.out}/bin":"$PATH"

                # TODO: only enable if python bindings enabled
                export PYTHONPATH="$PYTHONPATH":"$PWD"/build/python

                # Needed for using mlir-tblgen inside the dev shell
                export LD_LIBRARY_PATH=${pkgs.z3.lib}/lib:$LD_LIBRARY_PATH
              '';
            });
          };

          devShellBaseWithDefault = pkgs: final.devShellBase pkgs final.llzk-debug;
        };
        # So downstream users don't need to apply the pcl-mlir-pkg overlay directly
        overlays.default = nixpkgs.lib.composeExtensions pcl-mlir-pkg.overlays.default self.overlays.llzk;
    }
    // (flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;

          overlays = [
            self.overlays.default
            llzk-pkgs.overlays.default
            release-helpers.overlays.default
          ];
        };
      in
      {
        # Now, we can define the actual outputs of the flake
        packages = flake-utils.lib.flattenTree {
          # Copy the packages from imported overlays.
          inherit (pkgs) llzk llzk-debug;
          inherit (pkgs) mlir mlir-debug;
          inherit (pkgs) pcl-mlir pcl-mlir-debug;
          inherit (pkgs) changelogCreator;
          inherit (pkgs) mlir-with-python llzk-with-python;
          # Prevent use of libllvm and llvm from nixpkgs, which will have
          # different versions than mlir/llvm built here.
          inherit (pkgs.llzk-llvmPackages) libllvm llvm;

          default = pkgs.llzk;
          debugClang = pkgs.llzkDebWithSansClang;
          debugClangCov = pkgs.llzkDebWithSansClangCov;
          debugGCC = pkgs.llzkDebWithSansGCC;
          docs = pkgs.llzkDocs;
        };

        checks = flake-utils.lib.flattenTree {
          llzk-installcheck-release = pkgs.callPackage ./nix/llzk-installcheck {
            mlir_pkg = pkgs.mlir;
            llzk_pkg = pkgs.llzk;
            pcl_pkg = pkgs.pcl-mlir;
          };
          llzk-installcheck-debug = pkgs.callPackage ./nix/llzk-installcheck {
            mlir_pkg = pkgs.mlir-debug;
            llzk_pkg = pkgs.llzk-debug;
            pcl_pkg = pkgs.pcl-mlir-debug;
          };
        };

        devShells = flake-utils.lib.flattenTree {
          default = (pkgs.devShellBaseWithDefault pkgs).shell.overrideAttrs (_: {
            # Use Debug by default so assertions are enabled by default.
            cmakeBuildType = "Debug";
          });
          # Shell in release mode to test issues that may only appear in optimized builds
          release = (pkgs.devShellBase pkgs pkgs.llzk).shell.overrideAttrs (_: {
            cmakeBuildType = "Release";
          });
          debugClang = (pkgs.devShellBase pkgs pkgs.llzkDebWithSansClang).shell;
          debugGCC = (pkgs.devShellBase pkgs pkgs.llzkDebWithSansGCC).shell;
        };
      }
    ));
}
