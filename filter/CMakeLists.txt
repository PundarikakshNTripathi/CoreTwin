add_library(coretwin_filter STATIC filter_main.cpp)
target_include_directories(coretwin_filter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coretwin_filter PUBLIC Eigen3::Eigen)
