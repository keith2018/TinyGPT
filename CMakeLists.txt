cmake_minimum_required(VERSION 3.15)
project(TinyGPT)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(TINYGPT_BUILD_PYBINDING "Whether or not to build pybinding" OFF)
option(TINYGPT_BUILD_EXAMPLES "Whether or not to build examples" ON)
option(TINYGPT_BUILD_TEST "Whether or not to build the tests" ON)
option(TINYGPT_BUILD_SERVER "Whether or not to build the HTTP server" ON)

add_subdirectory(src)

if (${TINYGPT_BUILD_PYBINDING})
    set(TINYGPT_BUILD_TEST OFF)
    set(TINYGPT_BUILD_EXAMPLES OFF)
endif ()

if (${TINYGPT_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()

if (${TINYGPT_BUILD_SERVER})
    add_subdirectory(server)
endif ()

if (${TINYGPT_BUILD_TEST})
    enable_testing()
    add_subdirectory(test)
endif ()

message(STATUS "TINYGPT_BUILD_PYBINDING ${TINYGPT_BUILD_PYBINDING}")
message(STATUS "TINYGPT_BUILD_EXAMPLES ${TINYGPT_BUILD_EXAMPLES}")
message(STATUS "TINYGPT_BUILD_TEST ${TINYGPT_BUILD_TEST}")
message(STATUS "TINYGPT_BUILD_SERVER ${TINYGPT_BUILD_SERVER}")