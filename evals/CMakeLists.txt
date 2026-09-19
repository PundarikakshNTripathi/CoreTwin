add_executable(test_lbs test_lbs.cpp)
target_link_libraries(test_lbs PRIVATE coretwin_kernels)

add_executable(test_ik test_ik.cpp)
target_link_libraries(test_ik PRIVATE coretwin_filter Eigen3::Eigen)
