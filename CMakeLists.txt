cmake_minimum_required(VERSION 3.24)
project(CoreTwin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED) # for V4L2 fallback and image processing
find_package(Eigen3 REQUIRED)
# ONNX Runtime and TensorRT - typically found via system paths or custom modules
# We will leave these as optional/interface libraries for now

# Add subdirectories
add_subdirectory(capture)
add_subdirectory(inference)
add_subdirectory(kernels)
add_subdirectory(filter)
add_subdirectory(render)
add_subdirectory(evals)

# Configure compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()
