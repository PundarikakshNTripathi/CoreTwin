add_library(coretwin_inference STATIC inference_main.cpp)
target_include_directories(coretwin_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Will link TRT/ONNX later
