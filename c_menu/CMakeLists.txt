# Standalone CMake entry for the c_menu module.
#
# NOTE: Not wired into the main c_physics build yet.
# The intention is to either:
#  - add these sources into the existing geodesic_physics.dll target, OR
#  - build a separate gp_menu.dll and load it from Python.

cmake_minimum_required(VERSION 3.15)
project(c_menu LANGUAGES CXX)

# Export all symbols on Windows so the C ABI is visible without explicit __declspec.
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(c_menu SHARED
    menu_waveform.cpp
    menu_waveform_abi.cpp
    workbench_rows_abi.cpp
    table_abi.cpp
)

target_include_directories(c_menu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(c_menu PUBLIC cxx_std_17)
