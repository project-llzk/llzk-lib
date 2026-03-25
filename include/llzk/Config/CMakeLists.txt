# We do empty string comparisons to allow for 0 to be specified, as otherwise
# a "0" value is interpreted as false.
if("${CMAKE_PROJECT_VERSION_MAJOR}" STREQUAL "")
    message(FATAL_ERROR "Project must specify major version!")
endif()

if("${CMAKE_PROJECT_VERSION_MINOR}" STREQUAL "")
    message(FATAL_ERROR "Project must specify minor version!")
endif()

if("${CMAKE_PROJECT_VERSION_PATCH}" STREQUAL "")
    message(FATAL_ERROR "Project must specify patch version!")
endif()

# The tweak version is set by `cmake/GitVersion.cmake` as the number of commits since the version tag.
# LLZK doesn't use this for version numbering but we do present a warning during the build that the
# version is not a pure release version. In this case, the "0" value should be interpreted as false.
if("${CMAKE_PROJECT_VERSION_TWEAK}")
    message(WARNING
        "This build is using a commit that is ${CMAKE_PROJECT_VERSION_TWEAK} commits ahead of the last tagged release "
        "version (v${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH})."
    )
endif()

configure_file(Config.h.in Config.h)
