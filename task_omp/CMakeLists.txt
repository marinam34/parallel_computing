cmake_minimum_required(VERSION 3.16)
project(super_com LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)


set(CMAKE_C_COMPILER "/opt/homebrew/bin/gcc-15")
set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-15")


set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)

include_directories(${INC_DIR})

file(GLOB SOURCES ${SRC_DIR}/*.cpp)

add_library(super_com_lib ${SOURCES})

add_executable(super_com ${SRC_DIR}/main.cpp)
target_link_libraries(super_com PRIVATE super_com_lib)


find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    target_link_libraries(super_com_lib PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(super_com     PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING " OpenMP not found, check GCC or libomp installation.")
endif()


if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(super_com_lib PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(super_com_lib PRIVATE -O3 -Wall -Wextra -Xpreprocessor -fopenmp)
    target_link_libraries(super_com_lib PRIVATE -lomp)
endif()

message(STATUS " Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS " Source dir: ${SRC_DIR}")
message(STATUS " Include dir: ${INC_DIR}")
