cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(backprop_ninja)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCG_CXX_FLAGS}")

add_executable(backprop_ninja backprop_ninja.cpp
                            utils/utils.h)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} utils)
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(backprop_ninja "${TORCH_LIBRARIES}")
set_property(TARGET backprop_ninja PROPERTY CXX_STANDARD 17)
