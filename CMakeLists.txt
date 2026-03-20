cmake_minimum_required(VERSION 4.1.2)

project(stable-fluids VERSION 1.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# Toolchain
# ============================================================================
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

# ============================================================================
# Targets
# ============================================================================
add_library(
        stable-fluids
        SHARED
        stable-fluids.h
        backend-cuda.cu
        backend-cpu.cpp
        backend-parallel.cpp
)

add_executable(
        stable-fluids-app
        main.cu
)

target_compile_definitions(stable-fluids PRIVATE STABLE_FLUIDS_BUILD_SHARED)
target_include_directories(stable-fluids PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stable-fluids PRIVATE CUDA::cudart)

target_link_libraries(stable-fluids-app PRIVATE stable-fluids CUDA::cudart)

# ============================================================================
# Compile Options
# ============================================================================
if (MSVC)
    target_compile_options(stable-fluids PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    target_compile_options(stable-fluids-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
else ()
    target_compile_options(stable-fluids PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    target_compile_options(stable-fluids-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif ()

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

# ============================================================================
# CUDA Linking
# ============================================================================
set_target_properties(
        stable-fluids
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# Output Layout
# ============================================================================
foreach (target stable-fluids stable-fluids-app)
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
    foreach (target stable-fluids stable-fluids-app)
        set_target_properties(
                ${target}
                PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
        )
    endforeach ()
endif ()
