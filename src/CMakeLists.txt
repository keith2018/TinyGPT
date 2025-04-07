cmake_minimum_required(VERSION 3.10)
project(TinyGPT_lib)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_TEST OFF)
add_subdirectory(${THIRD_PARTY_DIR}/TinyTorch TinyTorch)

file(GLOB_RECURSE TinyGPT_src "./*.cpp" "./*.h")
add_library(${PROJECT_NAME}
        ${TinyGPT_src}
        ${THIRD_PARTY_DIR}/json11/json11.cpp
        ${RE2_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${THIRD_PARTY_DIR}/json11
        ${THIRD_PARTY_DIR}/TinyTorch
)

target_link_libraries(${PROJECT_NAME} TinyTorch)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
elseif (CMAKE_BUILD_TYPE STREQUAL Release)
    if (MSVC)
        set_source_files_properties(${TinyGPT_src} PROPERTIES COMPILE_FLAGS "/Ox")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif ()
endif ()
