cmake_minimum_required(VERSION 3.15)
project(TinyGPT_server)

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

add_executable(${PROJECT_NAME}
        HttpServer.cpp
        ApiHandler.cpp
        ChatTemplateUtils.cpp
        ServerUtils.cpp
        Main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${THIRD_PARTY_DIR}/TinyTorch/src
        ${THIRD_PARTY_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)
target_link_libraries(${PROJECT_NAME} TinyGPT_lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# copy web UI files to the output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/web
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/web
        COMMENT "Copying web UI files..."
)
