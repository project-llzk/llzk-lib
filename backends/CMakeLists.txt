# backends/CMakeLists.txt

add_subdirectory(r1cs)
add_subdirectory(zklean)
add_subdirectory(smt)
if(LLZK_WITH_PCL_BOOL)
  add_subdirectory(pcl-conv)
endif()
