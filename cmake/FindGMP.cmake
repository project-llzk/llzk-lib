find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARY NAMES gmp)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP REQUIRED_VARS GMP_LIBRARY GMP_INCLUDE_DIR)
mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)

if(GMP_FOUND AND NOT TARGET GMP::GMP)
  add_library(GMP::GMP UNKNOWN IMPORTED)
  set_target_properties(GMP::GMP PROPERTIES
    IMPORTED_LOCATION "${GMP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
endif()
