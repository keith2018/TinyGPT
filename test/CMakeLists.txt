cmake_minimum_required(VERSION 3.15)
project(TinyGPT_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest)

file(GLOB test_src
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(${PROJECT_NAME} ${test_src})

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${THIRD_PARTY_DIR}/TinyTorch
        ${THIRD_PARTY_DIR}
        googletest/googletest/include
        googletest/googlemock/include
)
target_link_libraries(${PROJECT_NAME} TinyGPT_lib gtest_main)

if (MSVC)
    set_source_files_properties(${test_src} PROPERTIES COMPILE_FLAGS "/utf-8")
endif ()

include(GoogleTest)

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

# add tests
gtest_discover_tests(${PROJECT_NAME} WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)
