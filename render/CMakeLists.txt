add_library(coretwin_render STATIC render_main.cpp)
target_include_directories(coretwin_render PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coretwin_render PUBLIC Vulkan::Vulkan)
