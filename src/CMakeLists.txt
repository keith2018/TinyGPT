cmake_minimum_required(VERSION 3.10)
project(TinyGPT_lib)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)

add_subdirectory(${THIRD_PARTY_DIR}/abseil-cpp abseil-cpp)
add_subdirectory(${THIRD_PARTY_DIR}/re2 re2)

file(GLOB_RECURSE TinyGPT_src "./*.cpp" "./*.h")

if (MSVC)
    set_source_files_properties(${TinyGPT_src} PROPERTIES COMPILE_FLAGS "/WX")
else ()
    set_source_files_properties(${TinyGPT_src} PROPERTIES COMPILE_FLAGS "-Werror -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()

add_library(${PROJECT_NAME}
        ${TinyGPT_src}
        ${THIRD_PARTY_DIR}/json11/json11.cpp
        ${RE2_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${THIRD_PARTY_DIR}/json11
        ${THIRD_PARTY_DIR}/re2
)

if (MSVC)
    set(MKL_ARCH intel64)
    set(MKL_LINK static)
    set(MKL_THREADING intel_thread)
    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME}
            re2::re2 MKL::MKL
    )
endif ()

if (APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(${PROJECT_NAME} re2::re2 ${ACCELERATE_LIBRARY})
endif ()
