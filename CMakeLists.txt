cmake_minimum_required(VERSION 3.15)
project(TinyGPT)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(TINYGPT_BUILD_PYBINDING "Whether or not to build pybinding" OFF)
option(TINYGPT_BUILD_DEMO "Whether or not to build demo" ON)
option(TINYGPT_BUILD_TEST "Whether or not to build the tests" ON)

add_subdirectory(src)

if (${TINYGPT_BUILD_PYBINDING})
    set(TINYGPT_BUILD_TEST OFF)
    set(TINYGPT_BUILD_DEMO OFF)
endif ()

if (${TINYGPT_BUILD_DEMO})
    add_subdirectory(app)
endif ()

if (${TINYGPT_BUILD_TEST})
    enable_testing()
    add_subdirectory(test)
endif ()

message(STATUS "TINYGPT_BUILD_PYBINDING ${TINYGPT_BUILD_PYBINDING}")
message(STATUS "TINYGPT_BUILD_DEMO ${TINYGPT_BUILD_DEMO}")
message(STATUS "TINYGPT_BUILD_TEST ${TINYGPT_BUILD_TEST}")