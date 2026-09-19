cmake_minimum_required(VERSION 3.15)
project(living_npc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/enginemath)
target_compile_definitions(enginemath INTERFACE ENGINEMATH_DEPTH_ZERO_TO_ONE)

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
file(GLOB SHADER_SOURCES
    ${SHADER_DIR}/*.vert
    ${SHADER_DIR}/*.frag
    ${SHADER_DIR}/*.comp)

set(SPV_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    add_custom_command(
        OUTPUT ${SHADER}.spv
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER} -o ${SHADER}.spv
        DEPENDS ${SHADER}
        COMMENT "compiling ${SHADER}")
    list(APPEND SPV_FILES ${SHADER}.spv)
endforeach()

add_custom_target(shaders DEPENDS ${SPV_FILES})

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_vulkan.cpp)

target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends)

target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

add_executable(living_npc
    src/main.cpp
    src/RenderingEngine/device.cpp
    src/RenderingEngine/Platform/window.cpp)

add_dependencies(living_npc shaders)

target_include_directories(living_npc PRIVATE src vendor)

target_link_libraries(living_npc PRIVATE
    enginemath::enginemath Vulkan::Vulkan glfw imgui)
