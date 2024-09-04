cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCG_CXX_FLAGS}")

add_executable(main backprop_ninja.cpp
                    utils/utils.h
                    utils/tools.cu)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} utils)
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(main "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 14)
