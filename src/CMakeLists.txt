cmake_minimum_required(VERSION 3.15)
project(TinyGPT_lib)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${TINYGPT_BUILD_PYBINDING})
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(BUILD_SHARED_LIBS OFF)
endif ()

set(TINYTORCH_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(TINYTORCH_BUILD_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/TinyTorch TinyTorch)

#pcre2
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2TEST OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/pcre2 pcre2)

file(GLOB TinyGPT_src
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp"
)

add_library(${PROJECT_NAME} STATIC ${TinyGPT_src})
target_link_libraries(${PROJECT_NAME} TinyTorch_lib pcre2-8)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${THIRD_PARTY_DIR}/pcre2/src
        ${THIRD_PARTY_DIR}/TinyTorch/src
        ${THIRD_PARTY_DIR}
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

if (${TINYGPT_BUILD_PYBINDING})
    add_subdirectory(${THIRD_PARTY_DIR}/pybind11 pybind11)
    pybind11_add_module(_tinygpt binding/pybind_tinygpt.cpp)
    target_include_directories(_tinygpt PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${THIRD_PARTY_DIR}/TinyTorch/src
            ${THIRD_PARTY_DIR}
            pybind11/include
    )
    target_link_libraries(_tinygpt PRIVATE TinyGPT_lib)
    set_target_properties(_tinygpt PROPERTIES PREFIX "" SUFFIX ".so")
    install(TARGETS _tinygpt LIBRARY DESTINATION tinygpt)
endif ()
