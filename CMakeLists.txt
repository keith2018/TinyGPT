cmake_minimum_required(VERSION 3.10)
project(TinyGPT)

add_subdirectory(src)
add_subdirectory(app)

option(TINY_GPT_BUILD_TEST "Whether or not to build the tests" ON)
if (${TINY_GPT_BUILD_TEST})
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()
