cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp-forntend)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCG_CXX_FLAGS}")

add_executable(cpp-forntend cpp-forntend.cpp
                            utils/utils.h)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} utils)
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(cpp-forntend "${TORCH_LIBRARIES}")
set_property(TARGET cpp-forntend PROPERTY CXX_STANDARD 17)
