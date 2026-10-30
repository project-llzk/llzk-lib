# Compile the registration source through MLIR's extension builder so it uses
# the final module's nanobind ABI, domain, and runtime settings.
add_library(LLZKPythonBindings INTERFACE)
target_link_libraries(LLZKPythonBindings INTERFACE LLZK::LLZKCAPI)
