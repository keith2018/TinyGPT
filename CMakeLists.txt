cmake_minimum_required(VERSION 3.10)
project(TinyGPT)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)
add_subdirectory(app)

option(BUILD_TEST "Whether or not to build the tests" ON)
if (${BUILD_TEST})
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()
