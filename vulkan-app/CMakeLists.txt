cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(stable-fluids-app VERSION 1.0.0 LANGUAGES CXX CUDA)


# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)
set(_VULKAN_HPP_IMPL "${CMAKE_CURRENT_BINARY_DIR}/_vulkan_hpp_impl.cpp")
file(WRITE "${_VULKAN_HPP_IMPL}" [[
#include <vulkan/vulkan_raii.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
]])
add_subdirectory(vendor/glfw)
set(imgui_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui")
set(_IMGUI_SOURCES
        vendor/imgui/imgui.cpp
        vendor/imgui/imgui_draw.cpp
        vendor/imgui/imgui_tables.cpp
        vendor/imgui/imgui_widgets.cpp
        vendor/imgui/backends/imgui_impl_glfw.cpp
        vendor/imgui/backends/imgui_impl_vulkan.cpp
)


# ============================================================================
# Library
# ============================================================================
add_library(vk-core STATIC)
add_library(vk-core::vk-core ALIAS vk-core)
target_sources(vk-core
        PRIVATE
        src/vk.camera.cpp
        src/vk.context.cpp
        src/vk.frame.cpp
        src/vk.geometry.cpp
        src/vk.imgui.cpp
        src/vk.math.cpp
        src/vk.memory.cpp
        src/vk.pipeline.cpp
        src/vk.swapchain.cpp
        src/vk.texture.cpp
        ${_IMGUI_SOURCES}
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        modules/vk.camera.ixx
        modules/vk.context.ixx
        modules/vk.frame.ixx
        modules/vk.geometry.ixx
        modules/vk.imgui.ixx
        modules/vk.math.ixx
        modules/vk.memory.ixx
        modules/vk.pipeline.ixx
        modules/vk.swapchain.ixx
        modules/vk.texture.ixx
)
target_link_libraries(vk-core PUBLIC Vulkan::Vulkan glfw)
target_compile_definitions(vk-core PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
target_include_directories(vk-core PUBLIC
        $<BUILD_INTERFACE:${imgui_SOURCE_DIR}>
)
set_property(TARGET vk-core PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Shader Compilation
# ============================================================================
find_program(
        SLANGC_EXECUTABLE
        NAMES slangc
        HINTS ENV VULKAN_SDK
        PATH_SUFFIXES Bin bin
)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SMOKE_SIM_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SMOKE_SIM_SHADER_SOURCES
        shaders/smoke_volume.slang
)
set(SMOKE_SIM_SHADER_OUTPUTS "")
foreach (shader IN LISTS SMOKE_SIM_SHADER_SOURCES)
    get_filename_component(shader_name "${shader}" NAME_WE)
    set(shader_output "${SMOKE_SIM_SHADER_DIR}/${shader_name}.spv")
    add_custom_command(
            OUTPUT "${shader_output}"
            COMMAND "${SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" -profile sm_6_5 -target spirv -entry vs_main -entry fs_main -o "${shader_output}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            COMMENT "Compiling ${shader_name}.slang"
            VERBATIM
    )
    list(APPEND SMOKE_SIM_SHADER_OUTPUTS "${shader_output}")
endforeach ()
add_custom_target(stable-fluids-shaders DEPENDS ${SMOKE_SIM_SHADER_OUTPUTS})


# ============================================================================
# Application
# ============================================================================
add_executable(stable-fluids-app)
target_sources(stable-fluids-app
        PRIVATE
        main.cpp
        app.cpp
        stable_simulation.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        app.ixx
)
add_dependencies(stable-fluids-app stable-fluids-shaders)
target_compile_definitions(stable-fluids-app PRIVATE SMOKE_SIM_SHADER_DIR="${SMOKE_SIM_SHADER_DIR}")
target_link_libraries(stable-fluids-app PRIVATE vk-core::vk-core stable-fluids CUDA::cudart)
set_property(TARGET stable-fluids-app PROPERTY CXX_MODULE_STD ON)