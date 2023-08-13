cmake_minimum_required(VERSION 3.10)
project(TinyGPT)

set(CMAKE_CXX_STANDARD 11)

set(MKL_ARCH intel64)
set(MKL_LINK static)
set(MKL_THREADING intel_thread)
find_package(MKL CONFIG REQUIRED)

add_subdirectory(src)
add_subdirectory(app)

option(BUILD_TEST "Whether or not to build the tests" ON)
if (${BUILD_TEST})
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()
