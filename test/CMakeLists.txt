cmake_minimum_required(VERSION 3.10)
project(TinyGPT_test)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest)

add_executable(${PROJECT_NAME}
        main.cpp
        test_tensor.cpp
        test_model.cpp
        test_tokenizer.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json11
        googletest/googlemock/include
)
target_link_libraries(${PROJECT_NAME} TinyGPT_lib gtest_main)

include(GoogleTest)

# copy windows dll
if (MSVC)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${MKL_ROOT}/../../compiler/latest/windows/redist/intel64_win/compiler/libiomp5md.dll
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/libiomp5md.dll
    )
endif()

# copy assets
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
)

# add tests
gtest_discover_tests(${PROJECT_NAME} WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)
