cmake_minimum_required(VERSION 4.1.2)

project(stable-fluids VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Toolchain
# ============================================================================
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
endif ()

set(STABLE_FLUIDS_WITH_CUDA OFF)
if (CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    set(STABLE_FLUIDS_WITH_CUDA ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif ()

# ============================================================================
# Targets
# ============================================================================
set(stable_fluids_sources
        stable-fluids.h
        validate.cpp
        backend-parallel.cpp
)

if (STABLE_FLUIDS_WITH_CUDA)
    list(APPEND stable_fluids_sources backend-cuda.cu)
endif ()

add_library(
        stable-fluids
        SHARED
        ${stable_fluids_sources}
)

if (STABLE_FLUIDS_WITH_CUDA)
    add_executable(
            stable-fluids-app
            main.cu
    )
endif ()

target_compile_definitions(stable-fluids PRIVATE STABLE_FLUIDS_BUILD_SHARED)
if (NOT STABLE_FLUIDS_WITH_CUDA)
    target_compile_definitions(stable-fluids PRIVATE STABLE_FLUIDS_NO_CUDA)
endif ()
target_include_directories(stable-fluids PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (STABLE_FLUIDS_WITH_CUDA)
    target_link_libraries(stable-fluids PRIVATE CUDA::cudart)
    target_link_libraries(stable-fluids-app PRIVATE stable-fluids CUDA::cudart)
endif ()

# ============================================================================
# Compile Options
# ============================================================================
if (MSVC)
    target_compile_options(stable-fluids PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    if (TARGET stable-fluids-app)
        target_compile_options(stable-fluids-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    endif ()
else ()
    target_compile_options(stable-fluids PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    if (TARGET stable-fluids-app)
        target_compile_options(stable-fluids-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    endif ()
endif ()

if (STABLE_FLUIDS_WITH_CUDA)
    target_compile_options(
            stable-fluids
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )
    target_compile_options(
            stable-fluids-app
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )

    set_target_properties(
            stable-fluids
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

# ============================================================================
# Output Layout
# ============================================================================
set(stable_fluids_targets stable-fluids)
if (TARGET stable-fluids-app)
    list(APPEND stable_fluids_targets stable-fluids-app)
endif ()

foreach (target IN LISTS stable_fluids_targets)
    set_target_properties(
            ${target}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach ()

# ============================================================================
# Runtime Lookup
# ============================================================================
if (UNIX)
    foreach (target IN LISTS stable_fluids_targets)
        set_target_properties(
                ${target}
                PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
        )
    endforeach ()
endif ()
