cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp-forntend)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCG_CXX_FLAGS}")

add_executable(cpp-forntend cpp-forntend.cpp)
target_link_libraries(cpp-forntend "${TORCH_LIBRARIES}")
set_property(TARGET cpp-forntend PROPERTY CXX_STANDARD 17)
