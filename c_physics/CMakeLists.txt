cmake_minimum_required(VERSION 3.20)

project(geodesic_physics C)

# Build a shared library that can be loaded from Python via ctypes.
add_library(geodesic_physics SHARED
    geodesic_physics.c
    weapon_sim.c
)

target_include_directories(geodesic_physics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(geodesic_physics PRIVATE c_std_11)

# Keep the runtime name stable for ctypes loading.
set_target_properties(geodesic_physics PROPERTIES
    OUTPUT_NAME "geodesic_physics"
)

if (MSVC)
    target_compile_options(geodesic_physics PRIVATE /W4)
else()
    target_compile_options(geodesic_physics PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(geodesic_physics PRIVATE m)
endif()
