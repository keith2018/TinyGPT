cmake_minimum_required(VERSION 3.15)
project(TinyGPT_demo)

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

add_executable(${PROJECT_NAME}
        gpt2.cpp
        tokenizer.cpp
        main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${THIRD_PARTY_DIR}/TinyTorch
        ${THIRD_PARTY_DIR}
)

target_link_libraries(${PROJECT_NAME} TinyGPT_lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

if (${USE_BLAS})
    # copy libopenblas.dll
    if (MSVC)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${THIRD_PARTY_DIR}/TinyTorch/thirdparty/OpenBLAS/win64-64/bin/libopenblas.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/libopenblas.dll
        )
    endif ()
endif ()

# copy assets
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
)
