cmake_minimum_required(VERSION 3.20)

# CMake 4.x policy: honor <PACKAGE>_ROOT variables
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
# CMake 4.x policy: FindBoost removed
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(grilly_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find Vulkan SDK ──────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")

# ── Boost 1.82 ───────────────────────────────────────────────────────────
# Header-only: container (flat_map), lockfree, pool
# Link-time: atomic (required by lockfree)
set(BOOST_ROOT "C:/local/boost_1_82_0" CACHE PATH "Boost installation root")
set(BOOST_LIBDIR "${BOOST_ROOT}/lib64-msvc-10.0")

if(NOT EXISTS "${BOOST_ROOT}/boost/version.hpp")
    message(FATAL_ERROR
        "Boost not found at ${BOOST_ROOT}\n"
        "Set -DBOOST_ROOT=/path/to/boost or install Boost 1.82+")
endif()

# Header-only target
add_library(Boost::headers INTERFACE IMPORTED)
set_target_properties(Boost::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BOOST_ROOT}"
)

# Boost.Atomic — find the release DLL import lib
file(GLOB _boost_atomic_lib "${BOOST_LIBDIR}/boost_atomic-*-mt-x64-1_82.lib")
if(_boost_atomic_lib)
    list(GET _boost_atomic_lib 0 _boost_atomic_lib)
    add_library(Boost::atomic SHARED IMPORTED)
    set_target_properties(Boost::atomic PROPERTIES
        IMPORTED_IMPLIB "${_boost_atomic_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${BOOST_ROOT}"
    )
    file(GLOB _boost_atomic_dll "${BOOST_LIBDIR}/boost_atomic-*-mt-x64-1_82.dll")
    if(_boost_atomic_dll)
        list(GET _boost_atomic_dll 0 _boost_atomic_dll)
        set_target_properties(Boost::atomic PROPERTIES
            IMPORTED_LOCATION "${_boost_atomic_dll}"
        )
    endif()
    message(STATUS "Boost.Atomic: ${_boost_atomic_lib}")
else()
    message(WARNING "Boost.Atomic lib not found — creating header-only alias")
    add_library(Boost::atomic ALIAS Boost::headers)
endif()
message(STATUS "Boost configured at ${BOOST_ROOT}")

# ── Eigen 5 (header-only) ───────────────────────────────────────────────
set(EIGEN_ROOT "C:/local/eigen-5.0.0" CACHE PATH "Eigen installation root")
list(APPEND CMAKE_PREFIX_PATH "${EIGEN_ROOT}" "${EIGEN_ROOT}/cmake")
find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 CMake config not found — using header path directly")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_ROOT}"
    )
else()
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
endif()

# ── VMA (header-only) ───────────────────────────────────────────────────
set(VMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/VulkanMemoryAllocator/include")
if(NOT EXISTS "${VMA_INCLUDE_DIR}/vk_mem_alloc.h")
    message(FATAL_ERROR
        "VMA not found at ${VMA_INCLUDE_DIR}/vk_mem_alloc.h\n"
        "Run: git submodule update --init")
endif()

# ── pybind11 ─────────────────────────────────────────────────────────────
# scikit-build-core provides pybind11 via build dependencies
set(PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11")
if(EXISTS "${PYBIND11_DIR}/CMakeLists.txt")
    add_subdirectory("${PYBIND11_DIR}" pybind11)
else()
    find_package(pybind11 REQUIRED)
endif()

# ── BLAKE3 (C library for CubeMind VSA role generation) ──────────────────
set(BLAKE3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/BLAKE3/c")
if(NOT EXISTS "${BLAKE3_DIR}/blake3.h")
    message(FATAL_ERROR
        "BLAKE3 not found at ${BLAKE3_DIR}\n"
        "Run: git submodule update --init")
endif()
add_library(blake3 STATIC
    "${BLAKE3_DIR}/blake3.c"
    "${BLAKE3_DIR}/blake3_dispatch.c"
    "${BLAKE3_DIR}/blake3_portable.c"
)
target_include_directories(blake3 PUBLIC "${BLAKE3_DIR}")
if(MSVC)
    target_sources(blake3 PRIVATE
        "${BLAKE3_DIR}/blake3_sse2.c"
        "${BLAKE3_DIR}/blake3_sse41.c"
        "${BLAKE3_DIR}/blake3_avx2.c"
        "${BLAKE3_DIR}/blake3_avx512.c"
    )
    target_compile_options(blake3 PRIVATE /wd4244 /wd4100 /wd4127)
else()
    target_sources(blake3 PRIVATE
        "${BLAKE3_DIR}/blake3_sse2.c"
        "${BLAKE3_DIR}/blake3_sse41.c"
        "${BLAKE3_DIR}/blake3_avx2.c"
        "${BLAKE3_DIR}/blake3_avx512.c"
    )
endif()
message(STATUS "BLAKE3 configured at ${BLAKE3_DIR}")

# ── Core static library ─────────────────────────────────────────────────
# Source paths updated for monorepo: cpp/src/ and cpp/include/
add_library(grilly_core_lib STATIC
    cpp/src/device.cpp
    cpp/src/buffer_pool.cpp
    cpp/src/pipeline_cache.cpp
    cpp/src/command_batch.cpp
    cpp/src/op_graph.cpp
    cpp/src/ops/linear.cpp
    cpp/src/ops/activations.cpp
    cpp/src/ops/layernorm.cpp
    cpp/src/ops/attention.cpp
    cpp/src/ops/conv.cpp
    cpp/src/ops/kv_cache.cpp
    cpp/src/ops/swizzle.cpp
    cpp/src/experimental/paged_latent_pool.cpp
    cpp/src/experimental/fused_attention.cpp
    cpp/src/cubemind/vsa.cpp
    cpp/src/cubemind/cube.cpp
    cpp/src/cubemind/cache.cpp
    cpp/src/cubemind/text_encoder.cpp
    cpp/src/cubemind/semantic_assigner.cpp
    cpp/src/cubemind/resonator.cpp
    cpp/src/training/pipeline.cpp
    cpp/src/cognitive/world_model.cpp
    cpp/src/temporal/vulkan_temporal.cpp
    cpp/src/autograd.cpp
)

target_include_directories(grilly_core_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${VMA_INCLUDE_DIR}"
)

target_link_libraries(grilly_core_lib PUBLIC
    Vulkan::Vulkan
    Boost::headers
    Boost::atomic
    Eigen3::Eigen
    blake3
)

# VMA implementation compiled in device.cpp
# Platform-specific flags
if(MSVC)
    target_compile_options(grilly_core_lib PRIVATE /W4 /permissive-)
    target_compile_options(grilly_core_lib PRIVATE /wd4127 /wd4244 /wd4702 /wd4100)
else()
    target_compile_options(grilly_core_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── Python module ────────────────────────────────────────────────────────
pybind11_add_module(grilly_core cpp/python/bindings.cpp)
target_link_libraries(grilly_core PRIVATE grilly_core_lib)

# Install the extension module into the grilly package
install(TARGETS grilly_core DESTINATION grilly)
