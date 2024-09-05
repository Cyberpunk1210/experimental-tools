cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCG_CXX_FLAGS}")

# cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;-G;-g)

add_executable(main backprop_ninja.cpp
                    utils/utils.h
                    utils/tools.cuh)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} utils)
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(main "${TORCH_LIBRARIES}")

set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++14")
# CUDA_ADD_EXECUTABLE(main)

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 14)
