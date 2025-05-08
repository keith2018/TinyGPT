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

# regex
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(${THIRD_PARTY_DIR}/abseil-cpp abseil-cpp)
add_subdirectory(${THIRD_PARTY_DIR}/re2 re2)

file(GLOB_RECURSE TinyGPT_src "*.cpp")

add_library(${PROJECT_NAME}
        ${TinyGPT_src}
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${THIRD_PARTY_DIR}/abseil-cpp
        ${THIRD_PARTY_DIR}/re2
        ${THIRD_PARTY_DIR}/TinyTorch
        ${THIRD_PARTY_DIR}
)

target_link_libraries(${PROJECT_NAME}
        TinyTorch
        re2::re2
)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
elseif (CMAKE_BUILD_TYPE STREQUAL Release)
    if (MSVC)
        set_source_files_properties(${TinyGPT_src} PROPERTIES COMPILE_FLAGS "/Ox")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    endif ()
endif ()
