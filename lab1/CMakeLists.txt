cmake_minimum_required(VERSION 3.6)
project(lab1)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

message(STATUS "adding lab1")
add_executable("lab1" "main.cpp" first/Vector.cpp)
target_link_libraries("lab1" ${MPI_C_LIBRARIES})