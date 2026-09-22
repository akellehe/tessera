cmake_minimum_required(VERSION 3.18)
option(TESSERA_CUDA "Build with CUDA GPU acceleration for the Regge solver" ON)
if (DEFINED ENV{TESSERA_CUDA} AND NOT "$ENV{TESSERA_CUDA}" STREQUAL "")
    set(TESSERA_CUDA $ENV{TESSERA_CUDA})
endif()

if (TESSERA_CUDA)
    # Help CMake find nvcc at the standard /usr/local/cuda path
    if (NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        project(tessera LANGUAGES CXX CUDA)
    else()
        message(WARNING "CUDA requested but nvcc not found — falling back to CPU.")
        set(TESSERA_CUDA OFF)
        project(tessera LANGUAGES CXX)
    endif()
else()
    project(tessera LANGUAGES CXX)
endif()

# ---- Build resource limits (memory-aware Ninja job pools) ----
# A full tessera build is memory-bound, not CPU-bound. The template-heavy
# translation units (src/cobordism/MultiCobordism.cpp, the per-subsystem
# Bindings.cpp, the Eigen-dense chainhodge sources) each peak near 4.7 GB in
# cc1plus at -O3 -march=native. Ninja's default parallelism is (cores + 2), so
# on a 16-core box a plain `pip install -e ".[dev]"` launches 18 of those at
# once — tens of GB of demand against whatever the machine actually has. The
# box then hits 0 available memory, the OOM killer reaps unrelated processes,
# and the half-dead build leaves orphaned multi-GB cc1plus behind that have to
# be reaped by hand.
#
# CMAKE_BUILD_PARALLEL_LEVEL does not fix this. It is per-invocation, so every
# entry point (pip, cmake --build, an IDE, a script) has to remember to set it;
# and it is a single number covering both compiles and links, which have very
# different footprints. Ninja job pools are baked into build.ninja at configure
# time, so every entry point inherits them, and compiles and links get separate
# budgets. A pool only ever caps parallelism *below* -j, so an over-ambitious
# CMAKE_BUILD_PARALLEL_LEVEL is now clamped rather than obeyed.
#
# The budget is real memory minus a reserve for the rest of the machine,
# divided by a conservative per-job peak, then capped by the core count so a
# big-memory box does not oversubscribe the CPU instead.
#
# Escape hatches: -DTESSERA_BUILD_JOB_POOLS=OFF restores unbounded parallelism;
# the three _MB knobs below retune the budget without editing this file.
option(TESSERA_BUILD_JOB_POOLS
    "Cap compile/link parallelism by available memory (Ninja generators only)" ON)
set(TESSERA_COMPILE_MEMORY_MB 3072 CACHE STRING
    "Assumed peak resident memory of one C++ compile job, in MiB")
set(TESSERA_LINK_MEMORY_MB 6144 CACHE STRING
    "Assumed peak resident memory of one link job, in MiB")
set(TESSERA_BUILD_MEMORY_RESERVE_MB 4096 CACHE STRING
    "Memory left free for the rest of the machine during a build, in MiB")

if (TESSERA_BUILD_JOB_POOLS AND CMAKE_GENERATOR MATCHES "Ninja")
    # TOTAL_PHYSICAL_MEMORY is reported in MiB, matching the knobs above.
    cmake_host_system_information(RESULT _tessera_total_mem_mb
        QUERY TOTAL_PHYSICAL_MEMORY)
    cmake_host_system_information(RESULT _tessera_cores
        QUERY NUMBER_OF_LOGICAL_CORES)
    if (NOT _tessera_cores OR _tessera_cores LESS 1)
        set(_tessera_cores 1)
    endif()

    # Never let the reserve consume the whole budget on a small machine: a
    # 2 GB box would otherwise compute a zero budget and get clamped to 1 job
    # anyway, but going through 0 would make the arithmetic below meaningless.
    if (_tessera_total_mem_mb GREATER TESSERA_BUILD_MEMORY_RESERVE_MB)
        math(EXPR _tessera_mem_budget_mb
            "${_tessera_total_mem_mb} - ${TESSERA_BUILD_MEMORY_RESERVE_MB}")
    else()
        set(_tessera_mem_budget_mb "${_tessera_total_mem_mb}")
    endif()

    math(EXPR _tessera_compile_jobs
        "${_tessera_mem_budget_mb} / ${TESSERA_COMPILE_MEMORY_MB}")
    math(EXPR _tessera_link_jobs
        "${_tessera_mem_budget_mb} / ${TESSERA_LINK_MEMORY_MB}")

    if (_tessera_compile_jobs LESS 1)
        set(_tessera_compile_jobs 1)
    endif()
    if (_tessera_compile_jobs GREATER _tessera_cores)
        set(_tessera_compile_jobs ${_tessera_cores})
    endif()
    # Links are heavier than compiles and there are 20+ of them (every
    # tests/quantum/*.cpp is its own executable linking libtessera_quantum.a),
    # so they get their own, smaller pool — but never more than the compile
    # pool, which is already the machine-wide ceiling.
    if (_tessera_link_jobs LESS 1)
        set(_tessera_link_jobs 1)
    endif()
    if (_tessera_link_jobs GREATER _tessera_compile_jobs)
        set(_tessera_link_jobs ${_tessera_compile_jobs})
    endif()

    set_property(GLOBAL APPEND PROPERTY JOB_POOLS
        tessera_compile=${_tessera_compile_jobs}
        tessera_link=${_tessera_link_jobs})
    # These initialise the JOB_POOL_COMPILE / JOB_POOL_LINK target properties,
    # so they must be set before any add_library/add_executable below — which
    # includes tessera_build_itensor()'s ~30 translation units.
    set(CMAKE_JOB_POOL_COMPILE tessera_compile)
    set(CMAKE_JOB_POOL_LINK tessera_link)

    message(STATUS
        "Build parallelism capped at ${_tessera_compile_jobs} compile / "
        "${_tessera_link_jobs} link jobs "
        "(${_tessera_total_mem_mb} MiB RAM, ${_tessera_cores} cores; "
        "-DTESSERA_BUILD_JOB_POOLS=OFF to disable).")
elseif (TESSERA_BUILD_JOB_POOLS)
    message(STATUS
        "Generator '${CMAKE_GENERATOR}' has no job pools — build parallelism "
        "is uncapped. Prefer Ninja, or pass a small CMAKE_BUILD_PARALLEL_LEVEL "
        "so the template-heavy translation units do not exhaust memory.")
endif()

# ---- Compiler cache (ccache / sccache) ----
# Content-addressed object cache: makes clean rebuilds and per-wheel-tag
# builds near-instant — in particular ITensor's ~35 translation units,
# whose objects hash identically across Python versions and build dirs.
# Auto-enabled when found; set TESSERA_CCACHE=0 to opt out.
if (NOT "$ENV{TESSERA_CCACHE}" STREQUAL "0")
    find_program(CCACHE_PROGRAM NAMES ccache sccache)
    if (CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        message(STATUS "Compiler cache enabled: ${CCACHE_PROGRAM}")
    else()
        message(STATUS "Compiler cache not found — install ccache for fast rebuilds.")
    endif()
endif()

# ---- Faster linker (mold / lld) ----
# Linking _tessera against ITensor + Eigen + a BLAS/LAPACK backend is link-heavy;
# mold and lld cut link time sharply versus the default BFD ld. Preferred order
# mold > lld. Auto-enabled when found; set TESSERA_FAST_LINKER=0 to opt out.
#
# Release is DELIBERATELY excluded. The Release module links with LTO
# (-flto=auto, enabled by pybind11) at -O3, and lld 18.1.3 silently emits an
# empty _tessera.so with no PyInit__tessera for that link, so `import tessera`
# fails (#212). The identical Release objects link correctly under the default
# BFD ld (and gold), and lld links Debug fine. CI is build-only and never
# imports, so it slipped through. mold + Release is untested, so it is treated
# as equally unsafe. The fast linker therefore only applies to non-Release
# configs (Debug etc.), where it both links and runs and dev builds want the
# speed; Release falls back to the system default linker. This is a
# single-config build (scikit-build sets CMAKE_BUILD_TYPE), so a plain
# CMAKE_BUILD_TYPE guard is sufficient. RelWithDebInfo is excluded for the
# same reason: it is the Release build plus DWARF (#682) — same LTO link,
# same lld/mold hazard.
if (NOT "$ENV{TESSERA_FAST_LINKER}" STREQUAL "0"
        AND NOT CMAKE_BUILD_TYPE STREQUAL "Release"
        AND NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    find_program(MOLD_PROGRAM mold)
    find_program(LLD_PROGRAM ld.lld)
    if (MOLD_PROGRAM)
        add_link_options(-fuse-ld=mold)
        message(STATUS "Fast linker enabled: mold")
    elseif (LLD_PROGRAM)
        add_link_options(-fuse-ld=lld)
        message(STATUS "Fast linker enabled: lld")
    else()
        message(STATUS "Fast linker not found — install lld or mold for faster links.")
    endif()
endif()

option(TESSERA_ASAN "Build tessera with AddressSanitizer/UBSan" $ENV{TESSERA_ASAN})
if(NOT TESSERA_ASAN)
    set(TESSERA_ASAN OFF)
endif()
option(TESSERA_VERBOSE "Build tessera with C++ level logging" $ENV{TESSERA_VERBOSE})
if(NOT TESSERA_VERBOSE)
    set(TESSERA_VERBOSE OFF)
endif()
option(TESSERA_ASSERTIONS "Build with assertions all over the place to encourage early failures." $ENV{TESSERA_ASSERTIONS})
if (NOT TESSERA_ASSERTIONS)
    set(TESSERA_ASSERTIONS OFF)
endif()
option(TESSERA_PROFILE "Build with -g -O1 to keep symbol names for stack traces" $ENV{TESSERA_PROFILE})
if (NOT TESSERA_PROFILE)
    set(TESSERA_PROFILE OFF)
endif()
# ---- Quantum subsystem (Schwinger / MPS / DMRG): always built ----
# ITensor and a BLAS/LAPACK backend are unconditional dependencies of
# tessera. The ITensor submodule under third_party/itensor is fetched
# automatically when missing (one-time network download); a missing
# BLAS/LAPACK backend is a hard error. Eigen3 (resolved below near ZLIB)
# is likewise unconditional and is linked into tessera_core so every
# subsystem — including cobordism — can use it.
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/itensor/itensor/itensor.h")
    find_package(Git QUIET)
    if (NOT Git_FOUND OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        message(FATAL_ERROR
            "third_party/itensor is not checked out and cannot be fetched "
            "automatically (git not found, or this is not a git checkout).\n"
            "  Install tessera from a git clone, or fetch it manually:\n"
            "    git submodule update --init --recursive")
    endif()
    message(STATUS "Fetching the ITensor submodule (one-time network download)...")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
                -- third_party/itensor
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _itensor_submodule_result)
    if (NOT _itensor_submodule_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to fetch the ITensor submodule (git exited "
            "${_itensor_submodule_result}).\n"
            "  Check your network, then re-run the build — or fetch it "
            "manually:\n"
            "    git submodule update --init --recursive")
    endif()
    if (NOT EXISTS
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/itensor/itensor/itensor.h")
        message(FATAL_ERROR
            "ITensor submodule fetch reported success but "
            "third_party/itensor is still empty.")
    endif()
    message(STATUS "ITensor submodule ready.")
endif()

# BLAS/LAPACK backend: Accelerate ships with macOS; elsewhere it is required.
if (NOT APPLE)
    find_package(LAPACK REQUIRED)
endif()
# ---- Read version from pyproject.toml ----
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENTS)
string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+[^\"]*)" _ "${PYPROJECT_CONTENTS}")
set(TESSERA_VERSION "${CMAKE_MATCH_1}")
message(STATUS "tessera version: ${TESSERA_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python & pybind11 ----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Locate pybind11 >= 3.0. tessera's bindings use the v3 API, so an older
# v2 must never be selected — and the distro's ``pybind11-dev`` installs a
# config into ``/usr/lib/cmake``, a default find_package() search path.
#
# The ``3.0`` floor is enforced *by* find_package: a too-old config is
# skipped, and if nothing satisfies it the build fails loudly instead of
# silently linking the wrong ABI. CMake's CONFIG-mode search order then
# covers both supported build paths without any manual pinning:
#   * ``pip install`` — scikit-build-core puts the pinned pybind11 (from
#     build-system.requires) on CMAKE_PREFIX_PATH; searched ahead of HINTS.
#   * direct ``cmake`` — the active interpreter's own pybind11, supplied as
#     the HINTS fallback below. Empty (and harmless) if none is installed.
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_hint
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
find_package(pybind11 3.0 CONFIG REQUIRED HINTS "${_pybind11_hint}")
message(STATUS "pybind11 ${pybind11_VERSION} (${pybind11_DIR})")
find_package(ZLIB REQUIRED)
# Eigen3 is an unconditional dependency (used by the quantum and cobordism
# subsystems). Resolved here so it can be linked into tessera_core below.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(CheckCXXSourceCompiles)

check_cxx_source_compiles("
  #include <syncstream>
  int main() {
      std::osyncstream out(std::cout);
      out << \"test\" << std::endl;
      return 0;
  }
" HAVE_STD_OSYNCSTREAM)

if (HAVE_STD_OSYNCSTREAM)
    add_compile_definitions(TESSERA_HAS_OSYNCSTREAM=1)
else()
    add_compile_definitions(TESSERA_HAS_OSYNCSTREAM=0)
endif()



# ---- <Define source partitions> ------------------------------------------
# Two libraries:
#   tessera_core  — pybind-free C++ implementation: mesh primitives,
#                 spacetime, observables, Poset infrastructure, etc.
#                 Linked into _tessera and (when enabled) tessera_quantum.
#   tessera_quantum — Schwinger / DMRG / TDVP code (always built),
#                 also pybind-free, links tessera_core PUBLIC for shared
#                 mesh / Poset types.
# Plus a thin pybind11 module:
#   _tessera       — bindings.cpp + quantum/bindings.cpp; links the two libs.
file(GLOB_RECURSE TESSERA_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

# Quantum lives in its own static lib; exclude its sources from tessera_core.
set(TESSERA_CORE_SOURCES ${TESSERA_SOURCES})
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/(src|include)/quantum/")
# The RL subsystem (libtorch) is its own static lib (tessera_rl) — keep it out of the
# torch-free core, exactly like quantum.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/(src|include)/rl/")
# InteractionSimulation lives in src/simulations/ for architectural
# alignment with CDT (#44), but depends on Eigen + ITensor and so belongs
# in tessera_quantum, not tessera_core.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/(src|include)/simulations/InteractionSimulation\\.(cpp|h)$")
# ColorFiber lives in src/observables/ per the spectral-fiber design-spec
# layout (#767), but delegates to the quantum exterior-algebra primitives
# (quantum/GradedFock.h, #766), so it belongs in tessera_quantum too.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/ColorFiber\\.cpp$")
# ExchangeHolonomy (#772) likewise: it lives in src/observables/ but its
# structural parities delegate to quantum/GradedFock.h (#766), so it
# compiles in tessera_quantum.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/ExchangeHolonomy\\.cpp$")
# FiberConnection (#770) likewise lives in src/observables/ but delegates its
# rank-three center/adjoint conventions to ColorFiber (#767), which compiles
# in tessera_quantum — so it follows ColorFiber there.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/FiberConnection\\.cpp$")
# ClusterRegister (#860) reads FiberConnection's transport leakage, so it
# includes FiberConnection.h and follows it into tessera_quantum.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/ClusterRegister\\.cpp$")
# SheetedColor (#1195) builds the exterior algebra of the sheet space from
# quantum/GradedFock.h (#766) and delegates its three-sheet sectors to
# ColorFiber (#767), so it compiles in tessera_quantum beside them.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/SheetedColor\\.cpp$")
# MonopoleSpin (#1196) wedges its determinants with the quantum exterior
# algebra (#766), so it follows the same route.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/MonopoleSpin\\.cpp$")
# ComplexTransport (#1199) composes its general-linear links with
# SheetedColor's ordered product and holonomy invariants (#1195), which compile
# in tessera_quantum, so it follows them there.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/src/observables/ComplexTransport\\.cpp$")
# bindings.cpp is the pybind11 entry point — goes into _tessera, not tessera_core.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/src/bindings\\.cpp$")
# Per-subsystem Bindings.cpp files also belong to _tessera, not tessera_core
# (they include pybind11 / Python.h).
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/src/[^/]+/Bindings\\.cpp$")

# Sources for the pybind11 module. The quantum bindings are always built.
set(TESSERA_PYBIND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mesh/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spacetime/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/simulations/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cobordism/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/chainhodge/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/quantum/Bindings.cpp")
# ---- </Define source partitions> -----------------------------------------

# ---- <Build CUDA kernels as static library> ----
if (TESSERA_CUDA)
    file(GLOB_RECURSE TESSERA_CUDA_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    )
    add_library(tessera_cuda STATIC ${TESSERA_CUDA_SOURCES})
    set_target_properties(tessera_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(tessera_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    find_package(CUDAToolkit REQUIRED)
    # cuBLAS (SGEMM) backs the FP32 r_U-gradient GPU path (eigenstate_cuda.cu).
    target_link_libraries(tessera_cuda PRIVATE CUDA::cudart CUDA::cublas)
endif()

# ---- <tessera_core static library> ----------------------------------------
# Holds all the C++ implementation that has no pybind dependency. Both
# the Python module (_tessera) and the optional tessera_quantum library link
# this so we don't compile the same .cpp's twice.
add_library(tessera_core STATIC ${TESSERA_CORE_SOURCES})
target_include_directories(tessera_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tessera_core PUBLIC ZLIB::ZLIB Eigen3::Eigen)

# Eigen is MPL-2.0, but a handful of its modules pull in LGPL/GPL code
# (unsupported/Eigen/FFT via FFTW, MPRealSupport via MPFR, and the
# SparseCholesky / OrderingMethods AMD routines). tessera only uses MPL-2.0
# modules (Core, Dense, SparseCore, KroneckerProduct); EIGEN_MPL2_ONLY makes
# any future `#include` of a non-MPL2 module a hard compile error rather than
# a silent copyleft contamination of the proprietary _tessera binary. PUBLIC
# so it reaches every consumer that compiles Eigen headers (tessera_quantum,
# the binding TUs, and the test executables all link tessera_core).
target_compile_definitions(tessera_core PUBLIC EIGEN_MPL2_ONLY)

# Optional OpenMP for the spectral-graph heat-kernel parallel loop and
# similar embarrassingly-parallel hot paths. Bit-identical to the
# serial code at OMP_NUM_THREADS=1 (the default in our 10-CPU-budget
# scan policy); only kicks in for solo / exploratory runs that set
# OMP_NUM_THREADS > 1 explicitly. Required so `#pragma omp parallel`
# compiles and links cleanly; if OpenMP is absent the pragmas are
# silently no-ops (we don't gate compilation on it).
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tessera_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found (${OpenMP_CXX_VERSION}); spectral-graph "
                   "heat-kernel will respect OMP_NUM_THREADS.")
else()
    message(STATUS "OpenMP not found; #pragma omp directives in "
                   "tessera_core will compile as no-ops.")
endif()
# tessera_core is pybind-free: the binding code (which uses pybind11) lives
# only in src/bindings.cpp and src/quantum/Bindings.cpp, both compiled
# directly into the _tessera Python module. Earlier versions of tessera had
# stray `#include <pybind11/pybind11.h>` in mesh/spacetime headers that
# served no purpose; those have been removed so tessera_core no longer
# needs Python at link time. Test executables linking tessera_quantum →
# tessera_core get clean C++ symbols only.
set_target_properties(tessera_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

# Compile definitions that must reach consumers (so #ifdef SOURCES_ROOT,
# TESSERA_VERSION, TESSERA_VERBOSE, etc. work in the Python bindings TU and
# in test executables).
target_compile_definitions(tessera_core PUBLIC
    SOURCES_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    TESSERA_VERSION="${TESSERA_VERSION}")

if (TESSERA_CUDA)
    message(STATUS "CUDA acceleration turned ON.")
    target_compile_definitions(tessera_core PUBLIC TESSERA_CUDA=1)
    target_link_libraries(tessera_core PUBLIC tessera_cuda CUDA::cudart CUDA::cublas)
else()
    message(STATUS "CUDA acceleration turned OFF.")
endif()

if(TESSERA_ASAN)
    message(STATUS "ASAN/UBSAN support ON")
    target_compile_options(tessera_core PUBLIC
            -O1 -g3
            -fno-omit-frame-pointer
            -fsanitize=address,undefined
            -fno-sanitize-recover=address,undefined
            -fno-optimize-sibling-calls)
    target_link_options(tessera_core PUBLIC
            -fsanitize=address,undefined)
    target_compile_definitions(tessera_core PUBLIC ASAN_ENABLED=1)
else()
    message(STATUS "ASAN/UBSAN support OFF")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR TESSERA_PROFILE)
    message(STATUS "DEBUGGING (-O1, -g3) Support ON.")
    target_compile_options(tessera_core PRIVATE
        -O1 -g3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # -g adds DWARF so perf/addr2line can name static functions; it changes
    # nothing about code generation, so runtime performance is identical to
    # the plain -O3 build. The shipped build type is RelWithDebInfo because
    # pybind11_add_module post-build STRIPS Release modules (keeping its LTO
    # extras) but exempts RelWithDebInfo — Release-with-symbols is impossible
    # without giving up the pybind11 extras (#682). The -O3 here overrides the
    # -O2 that CMake's default RelWithDebInfo flags would otherwise apply.
    message(STATUS "DEBUGGING Support OFF, will use -O3 -g (symbols kept).")
    target_compile_options(tessera_core PRIVATE -O3 -g)
endif()

if (TESSERA_VERBOSE)
    message(STATUS "Logging turned ON.")
    target_compile_definitions(tessera_core PUBLIC TESSERA_VERBOSE=1)
else()
    message(STATUS "Logging turned OFF.")
endif()

if (TESSERA_ASSERTIONS)
    message(STATUS "Assertions turned ON.")
    target_compile_definitions(tessera_core PUBLIC TESSERA_ASSERTIONS=1)
else()
    message(STATUS "Assertions turned OFF.")
endif()
# ---- </tessera_core static library> --------------------------------------

# ---- <Build the Python module> ----
pybind11_add_module(_tessera MODULE ${TESSERA_PYBIND_SOURCES})
target_link_libraries(_tessera PRIVATE tessera_core)

set_target_properties(_tessera PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
# ---- </Build the Python module> ----

# Install the C extension inside the tessera/ package directory
install(TARGETS _tessera
        LIBRARY DESTINATION tessera
        RUNTIME DESTINATION tessera
        ARCHIVE DESTINATION tessera
        BUNDLE  DESTINATION tessera
)

# ---- <Quantum subsystem (ITensor + Schwinger MPO)> ----
message(STATUS "Building quantum subsystem (ITensor + DMRG).")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildITensor.cmake)
tessera_build_itensor()

file(GLOB_RECURSE TESSERA_QUANTUM_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/quantum/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/quantum/*.hpp")
# InteractionSimulation lives in src/simulations/ for architectural
# alignment with CDT (#44) but its Eigen/ITensor dependency makes it
# part of tessera_quantum, not tessera_core.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/simulations/InteractionSimulation.cpp")
# ColorFiber (#767) lives in src/observables/ per the design-spec layout but
# delegates to quantum/GradedFock.h (#766), so it compiles here.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/ColorFiber.cpp")
# ExchangeHolonomy (#772): same layout, same #766 delegation.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/ExchangeHolonomy.cpp")
# FiberConnection (#770) delegates its rank-three center/adjoint conventions
# to ColorFiber (#767), so it compiles beside it.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/FiberConnection.cpp")
# ClusterRegister (#860) assembles the whitepaper's fiber acceptance and reads
# FiberConnection's leakage, so it compiles beside it.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/ClusterRegister.cpp")
# SheetedColor (#1195) carries colour as sheet multiplicity: it builds the
# exterior algebra of the sheet space from quantum/GradedFock.h (#766) and its
# three-sheet sectors from ColorFiber (#767), so it compiles here.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/SheetedColor.cpp")
# MonopoleSpin (#1196) builds its determinants and its superpositions with the
# quantum exterior algebra (#766), so it compiles here as well.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/MonopoleSpin.cpp")
# ComplexTransport (#1199) carries the complex fibre transport of matched Riesz
# frames and composes its links with SheetedColor's ordered product and
# holonomy invariants (#1195), so it compiles beside them.
list(APPEND TESSERA_QUANTUM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/ComplexTransport.cpp")
# bindings.cpp uses pybind11 + Python.h and goes into _tessera, not the
# pybind-free static lib.
list(FILTER TESSERA_QUANTUM_SOURCES EXCLUDE REGEX "/src/quantum/Bindings\\.cpp$")
add_library(tessera_quantum STATIC ${TESSERA_QUANTUM_SOURCES})
target_include_directories(tessera_quantum PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
# PUBLIC link to tessera_core: tessera::Poset / Vertex / EdgeList live in
# tessera_core, and quantum's Poset-using headers expose them in their
# public API. PUBLIC propagates symbols to test executables too.
# (Eigen3 is already linked transitively via tessera_core, but kept explicit
# here for clarity since quantum headers use it directly.)
target_link_libraries(tessera_quantum PUBLIC itensor Eigen3::Eigen tessera_core)
set_target_properties(tessera_quantum PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)

enable_testing()
file(GLOB TESSERA_QUANTUM_TESTS
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantum/*.cpp")
foreach(_test_src ${TESSERA_QUANTUM_TESTS})
    get_filename_component(_test_name "${_test_src}" NAME_WE)
    add_executable(${_test_name} "${_test_src}")
    # tessera_quantum PUBLIC-links tessera_core, so tests get its symbols
    # transitively without an explicit tessera_core link here.
    target_link_libraries(${_test_name} PRIVATE tessera_quantum)
    set_target_properties(${_test_name} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON)
    add_test(NAME ${_test_name} COMMAND ${_test_name})
endforeach()

# Wire the quantum bindings TU (added to TESSERA_PYBIND_SOURCES above)
# into _tessera by linking against the implementation library.
target_link_libraries(_tessera PRIVATE tessera_quantum)
# ---- </Quantum subsystem> ----

# ---- <RL subsystem (libtorch PPO) — optional, #551> ----
# The RL module links libtorch, so it is built ONLY when torch is importable in the build
# interpreter (the `rl` extra). The core `_tessera` stays libtorch-free. torch 2.x uses the
# CXX11 ABI (matches gcc's default + tessera_core), and the imported `torch` target carries
# TORCH_CXX_FLAGS + the libtorch include/link directories.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _torch_cmake_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _torch_probe)
if(_torch_probe EQUAL 0 AND _torch_cmake_prefix)
    # `_torch_cmake_prefix` is `<pip-torch>/share/cmake`; two levels up is the pip torch root,
    # whose `lib/` holds libtorch/libc10. Prepend it to CMAKE_PREFIX_PATH so the find_library()
    # calls inside TorchConfig.cmake resolve to THIS libtorch — not a conflicting Homebrew
    # `pytorch` keg, whose `libtorch.dylib` lives in the default search path /opt/homebrew/lib.
    # Without this, find_package can report the right version (pip's config/headers) while
    # linking Homebrew's older dylibs, and `import tessera._tessera_rl` then dies at load time
    # on a missing libtorch symbol (e.g. at::detail::getXLAHooks).
    get_filename_component(_torch_root "${_torch_cmake_prefix}/../.." ABSOLUTE)
    list(PREPEND CMAKE_PREFIX_PATH "${_torch_root}")
    # Align Torch's CUDA detection with the core's toolkit. Torch's Caffe2 cuda.cmake uses the
    # LEGACY FindCUDA module, which locates nvcc via PATH / the CUDA_HOME env var rather than
    # CMAKE_CUDA_COMPILER. On a box with more than one CUDA toolkit (e.g. /usr/local/cuda plus
    # an NVIDIA HPC SDK, whose bin/ is often first on PATH and whose CUDA_HOME is exported),
    # FindCUDA can select a different toolkit than find_package(CUDAToolkit) did and then
    # FATAL-error on an nvcc-vs-header version mismatch (pytorch/pytorch#8092) — which aborts
    # the whole configure, not just the RL subsystem. Pin the legacy module to the exact nvcc
    # the core compiles with so the two always agree; a CUDA-less box (macOS, most CI) skips
    # this and torch's CPU path is used.
    if(TESSERA_CUDA AND CMAKE_CUDA_COMPILER)
        get_filename_component(_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(_cuda_root "${_cuda_bin}" DIRECTORY)
        set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_root}" CACHE PATH "" FORCE)
        set(CUDA_NVCC_EXECUTABLE "${CMAKE_CUDA_COMPILER}" CACHE FILEPATH "" FORCE)
        message(STATUS "RL: pinned Torch's legacy FindCUDA to ${_cuda_root} (core's toolkit).")
    endif()
    find_package(Torch QUIET HINTS "${_torch_cmake_prefix}")
endif()
if(Torch_FOUND)
    message(STATUS "Building RL subsystem (libtorch ${Torch_VERSION}).")
    # tessera_rl: the RL harness (env, PPO, training) — it DRIVES MultiCobordism + ProtonSynthesis
    # (the source of truth) and never reimplements construction. Its own static lib, kept out
    # of the torch-free tessera_core (rl/ is excluded from TESSERA_CORE_SOURCES above),
    # mirroring tessera_quantum.
    file(GLOB TESSERA_RL_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/rl/*.cpp")
    list(FILTER TESSERA_RL_SOURCES EXCLUDE REGEX "/src/rl/Bindings\\.cpp$")
    add_library(tessera_rl STATIC ${TESSERA_RL_SOURCES})
    target_include_directories(tessera_rl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    # Put THIS libtorch's headers at the front of the search path. tessera_core links
    # Homebrew deps (Eigen/BLAS) that carry a bare `-isystem /opt/homebrew/include`, and if
    # Homebrew also has a `pytorch` keg installed, `/opt/homebrew/include/torch` shadows the
    # pip libtorch we are actually building against. Inherited (dependency) include dirs are
    # emitted after a target's own dirs, and `torch` is linked after `tessera_core`, so
    # without this the Homebrew copy of e.g. `torch/ordered_dict.h` gets pulled in first and
    # collides with the pip copy ("redefinition of 'OrderedDict'"). SYSTEM BEFORE pins the
    # pip torch include dirs ahead of everything so `<torch/...>` always resolves to one tree.
    target_include_directories(tessera_rl SYSTEM BEFORE PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(tessera_rl PUBLIC tessera_core torch)
    set_target_properties(tessera_rl PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON)

    pybind11_add_module(_tessera_rl MODULE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/rl/Bindings.cpp)
    # Same libtorch-header precedence fix as tessera_rl above: Bindings.cpp includes
    # <torch/torch.h> directly, so pin the pip torch dirs ahead of the Homebrew include dir
    # that tessera_core drags in transitively.
    target_include_directories(_tessera_rl SYSTEM BEFORE PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(_tessera_rl PRIVATE tessera_rl)
    # RPATH to libtorch's lib so `import tessera._tessera_rl` resolves libtorch_cpu.so without
    # importing torch first (the tessera.rl facade also imports torch first — belt + suspenders).
    set_target_properties(_tessera_rl PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF
            INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
            BUILD_WITH_INSTALL_RPATH ON)
    install(TARGETS _tessera_rl
            LIBRARY DESTINATION tessera
            RUNTIME DESTINATION tessera
            ARCHIVE DESTINATION tessera)
else()
    message(STATUS "torch not found — skipping the optional RL subsystem (_tessera_rl).")
endif()
# ---- </RL subsystem> ----

if(NOT TESSERA_ASAN)
# --- Generate Python stubs for tessera (.pyi) ---
set(STUB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")

add_custom_command(
        TARGET _tessera POST_BUILD
        BYPRODUCTS "${STUB_OUT_DIR}/_tessera.pyi"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${STUB_OUT_DIR}"
        # Make the just-built module importable:
        COMMAND ${Python_EXECUTABLE} -c
            "import pybind11_stubgen" && ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:_tessera>
            ${Python_EXECUTABLE} -m pybind11_stubgen _tessera -o "${STUB_OUT_DIR}"
                --ignore-invalid-expressions="tessera::.*"
            || ${CMAKE_COMMAND} -E echo "Warning: pybind11_stubgen not found, skipping stub generation"
        COMMENT "Generating pybind11 stubs -> ${STUB_OUT_DIR}/_tessera.pyi"
)

# Install the stub next to the extension in the package
install(FILES "${STUB_OUT_DIR}/_tessera.pyi" DESTINATION tessera OPTIONAL)
endif()
