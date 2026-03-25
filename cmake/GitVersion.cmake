# Fetch the project version from a git version tag.
# Note: if building in a sandbox where git is unavailable (e.g. Nix), pass
# -DLLZK_VERSION_OVERRIDE=<version> to CMake to bypass the call to this function.
function(get_git_version GIT_VERSION_VAR)
  execute_process(
    COMMAND git describe --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_VERSION)
    # The initial commit to the repository is tagged with `v0.0.0` so this will never fail except
    # in a nix sandbox environment where the version must be passed with `LLZK_VERSION_OVERRIDE`.
    message(FATAL_ERROR "Could not find git version tag and `LLZK_VERSION_OVERRIDE` is not set.")
  else()
    # The output of `git describe --tags` is in the format `vX.Y.Z-N-gHASH` where `vX.Y.Z` is the last tag,
    # `N` is the number of commits since the last tag, and `HASH` is the abbreviated commit hash. We need to
    # convert this to `X.Y.Z.N` for the LLZK version number.
    string(REGEX REPLACE "^v?([0-9]+\\.[0-9]+\\.[0-9]+)(-rc[0-9]+)?-([0-9]+)-g[0-9a-f]+" "\\1.\\3" VERSION_NUMBER "${GIT_VERSION}")
    # If the current commit is exactly at a tag, the output of `git describe --tags` is just `vX.Y.Z` which
    # is not matched by the previous case so handle that as well.
    string(REGEX REPLACE "^v" "" VERSION_NUMBER "${VERSION_NUMBER}")
    set(${GIT_VERSION_VAR} "${VERSION_NUMBER}" PARENT_SCOPE)
  endif()
endfunction()
