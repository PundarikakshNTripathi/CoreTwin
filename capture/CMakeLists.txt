add_library(coretwin_capture STATIC capture_main.cpp)
target_include_directories(coretwin_capture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coretwin_capture PUBLIC ${OpenCV_LIBS})
