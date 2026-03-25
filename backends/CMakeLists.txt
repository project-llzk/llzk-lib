# backends/CMakeLists.txt

add_subdirectory(r1cs)
add_subdirectory(zklean)
if(LLZK_WITH_PCL_BOOL)
  add_subdirectory(pcl-conv)
endif()
