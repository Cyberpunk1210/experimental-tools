cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main backprop_ninja.cpp
                    include/utils.h
                    ${ALL_SRCS})
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} utils)
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(main "${TORCH_LIBRARIES}" -lgomp)

# CUDA_ADD_EXECUTABLE(main)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

