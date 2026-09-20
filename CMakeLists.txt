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

foreach(SPV ${SPV_FILES})
    get_filename_component(SPV_NAME ${SPV} NAME)
    add_custom_command(TARGET shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV} ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME})
endforeach()
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

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
    src/RenderingEngine/Platform/window.cpp
    src/RenderingEngine/Helpers/buffer.cpp
    src/RenderingEngine/Helpers/image.cpp
    src/RenderingEngine/pipeline.cpp
    src/RenderingEngine/drawing.cpp
    src/RenderingEngine/renderer.cpp
    src/RenderingEngine/descriptor.cpp
    src/RenderingEngine/Model/model.cpp
    src/RenderingEngine/Model/uniform.cpp
    src/RenderingEngine/Model/texture.cpp
    src/RenderingEngine/Model/modelLoading.cpp
    src/RenderingEngine/Core/camera.cpp
    src/RenderingEngine/Core/input.cpp
    src/RenderingEngine/Core/cameraControls.cpp
    src/RenderingEngine/Utility/raycast.cpp
    src/ProceduralAnimation/faceAnimator.cpp
    src/outputText/parseOutput.cpp
    src/speechListener.cpp
    src/httpClient.cpp
    src/serverLauncher.cpp)

add_custom_command(TARGET living_npc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_BINARY_DIR}/models)

add_dependencies(living_npc shaders)

target_include_directories(living_npc PRIVATE src vendor)

target_compile_definitions(living_npc PRIVATE
    CONVO_LOG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/game_agents/data/conversation_log"
    PYTHON_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(living_npc PRIVATE
    enginemath::enginemath Vulkan::Vulkan glfw imgui)

# Azure Speech SDK (TTS + blendshape visemes)
set(SPEECHSDK ${CMAKE_CURRENT_SOURCE_DIR}/external/speechsdk)
target_include_directories(living_npc PRIVATE
    ${SPEECHSDK}/build/native/include/cxx_api
    ${SPEECHSDK}/build/native/include/c_api)
target_link_libraries(living_npc
    PRIVATE ${SPEECHSDK}/build/native/x64/Release/Microsoft.CognitiveServices.Speech.core.lib)
file(GLOB SPEECH_DLLS ${SPEECHSDK}/runtimes/win-x64/native/*.dll)
if(SPEECH_DLLS)
    add_custom_command(TARGET living_npc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPEECH_DLLS} $<TARGET_FILE_DIR:living_npc>
        COMMENT "copying Speech SDK dlls next to exe")
endif()
