# backends/CMakeLists.txt

add_subdirectory(r1cs)
if(LLZK_WITH_PCL_BOOL)
  add_subdirectory(pcl-conv)
endif()
