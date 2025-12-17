# Standalone CMake entry for the c_menu module.
#
# NOTE: Not wired into the main c_physics build yet.
# The intention is to either:
#  - add these sources into the existing geodesic_physics.dll target, OR
#  - build a separate gp_menu.dll and load it from Python.

cmake_minimum_required(VERSION 3.15)
project(c_menu LANGUAGES CXX)

include(CTest)

# Export all symbols on Windows so the C ABI is visible without explicit __declspec.
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(c_menu SHARED
    menu_waveform.cpp
    menu_waveform_abi.cpp
    workbench_rows_abi.cpp
    table_abi.cpp
    table_node_groups.cpp
    rope_sim.cpp
    text_render_helper.cpp
    canvas_abi.cpp
)

option(ENABLE_STB_EASY_FONT "Download and enable stb_easy_font.h for text helpers" ON)

if(ENABLE_STB_EASY_FONT)
    # download raw header at configure time into build tree
    file(DOWNLOAD
        https://raw.githubusercontent.com/nothings/stb/master/stb_easy_font.h
        ${CMAKE_CURRENT_BINARY_DIR}/stb_easy_font.h
        SHOW_PROGRESS
        STATUS _stb_dl_status
    )
    list(GET _stb_dl_status 0 _stb_dl_code)
    if(NOT _stb_dl_code EQUAL 0)
        message(WARNING "Failed to download stb_easy_font.h; continuing without it")
    else()
        target_include_directories(c_menu PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    endif()
else()
    target_include_directories(c_menu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

target_compile_features(c_menu PUBLIC cxx_std_17)

## Eigen acceleration: enabled by default (always attempt to fetch)
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
if(EXISTS ${eigen_SOURCE_DIR})
    # Newer Eigen provides Eigen3::Eigen target; otherwise include headers directly
    if (TARGET Eigen3::Eigen)
        target_link_libraries(c_menu PUBLIC Eigen3::Eigen)
        target_compile_definitions(c_menu PUBLIC EIGEN_SIM)
    else()
        # add include dir and still define EIGEN_SIM so code can try using Eigen headers
        target_include_directories(c_menu PUBLIC ${eigen_SOURCE_DIR})
        target_compile_definitions(c_menu PUBLIC EIGEN_SIM)
    endif()
else()
    message(WARNING "Failed to fetch Eigen; continuing without Eigen acceleration")
endif()

add_executable(canvas_table_headless_test
    tests/canvas_table_headless_test.cpp
)
target_link_libraries(canvas_table_headless_test PRIVATE c_menu)
target_include_directories(canvas_table_headless_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME canvas_table_headless_test COMMAND canvas_table_headless_test)
