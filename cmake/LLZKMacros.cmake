macro(llzk_target_add_mlir_link_settings target)
  llvm_update_compile_flags(${target})
  mlir_check_all_link_libraries(${target})
  # Allow aggregate consumers (notably Python extensions) to supply MLIR from
  # one shared runtime instead of embedding another copy of its static archives.
  get_target_property(llzk_link_libraries ${target} INTERFACE_LINK_LIBRARIES)
  if(llzk_link_libraries)
    set(llzk_filtered_libraries)
    foreach(llzk_link_library IN LISTS llzk_link_libraries)
      # Leave project targets unwrapped so CMake qualifies their exported names.
      # Private dependencies of static libraries are wrapped in LINK_ONLY.
      string(REGEX REPLACE "^[$]<LINK_ONLY:([^>]+)>$" "\\1"
             llzk_link_target "${llzk_link_library}")
      set(llzk_filtered_library "${llzk_link_library}")
      if(llzk_link_target MATCHES "^MLIR" AND TARGET ${llzk_link_target})
        get_target_property(llzk_imported ${llzk_link_target} IMPORTED)
        get_target_property(llzk_library_type ${llzk_link_target} TYPE)
        if(llzk_imported AND llzk_library_type STREQUAL "STATIC_LIBRARY")
          get_mlir_filtered_link_libraries(llzk_filtered_library ${llzk_link_library})
        endif()
      endif()
      list(APPEND llzk_filtered_libraries "${llzk_filtered_library}")
    endforeach()
    set_property(TARGET ${target} PROPERTY INTERFACE_LINK_LIBRARIES
                 "${llzk_filtered_libraries}")
  endif()
endmacro()

function(llzk_add_mlir_doc target_name out_filepath tblgen_flags)
  # this is a modified version of add_mlir_doc from AddMLIR.cmake
  set(OUT_FILE "${LLZK_MLIR_DOC_OUTPUT_DIR}/${out_filepath}")
  # ensure the file path's parent directories are created
  cmake_path(GET out_filepath PARENT_PATH OUT_PATH)
  file(MAKE_DIRECTORY "${LLZK_MLIR_DOC_OUTPUT_DIR}/${OUT_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/${OUT_PATH}")
  tablegen(MLIR ${out_filepath} ${tblgen_flags} ${ARGN})
  add_custom_command(
    OUTPUT ${OUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/${out_filepath}" "${OUT_FILE}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${out_filepath}")
  add_custom_target(${target_name} DEPENDS ${OUT_FILE})
  add_dependencies(mlir-doc ${target_name})
endfunction()

function(llzk_tablegen ofn)
  tablegen(LLZK ${ARGV})
  
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append the includes used for this file to the tablegen_compile_commands file.
  file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
      "--- !FileInfo:\n"
      "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
      "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
  )
endfunction()
