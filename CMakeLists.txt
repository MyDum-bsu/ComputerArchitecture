cmake_minimum_required(VERSION 3.6)
project(labs)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(labs "lab2 lab1")
separate_arguments(labs)

foreach(lab IN LISTS labs)
    set(lab_name "${lab}")
    message(STATUS "adding ${lab_name}")
    add_executable("${lab_name}" "${lab}/main.cpp")
    target_link_libraries("${lab_name}" ${MPI_C_LIBRARIES})
endforeach()

