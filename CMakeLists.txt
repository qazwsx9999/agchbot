cmake_minimum_required(VERSION 3.18)
project(alphazero_chess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W3 /openmp)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -pthread)
endif()

# Find LibTorch (PyTorch C++ Library)
# Usage: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Ensure dynamic linker finds Torch libraries at runtime on Linux
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/board.cpp
    src/action_encoding.cpp
    src/model.cpp
    src/mcts.cpp
    src/parallel_mcts.cpp
    src/replay_buffer.cpp
    src/trainer.cpp
    src/bots.cpp
    src/evaluate.cpp
    src/self_play.cpp
    src/main.cpp
)

# Executable target
add_executable(alphazero_chess ${SOURCES})

# Link LibTorch
target_link_libraries(alphazero_chess "${TORCH_LIBRARIES}")

# Checkpoint import tool: turns a converted Python checkpoint (weights.bin +
# meta.txt from tools/convert_checkpoint.py) into a native C++ checkpoint.pt.
add_executable(import_ckpt tools/import_ckpt.cpp src/model.cpp)
target_link_libraries(import_ckpt "${TORCH_LIBRARIES}")

# On MSVC + CUDA, torch's find_package enables the CUDA language, which makes
# the CUDA build customizations inject a bare "cudart_static.lib" into the
# link line. Put the CUDA toolkit lib dir on the linker search path so it
# resolves (LNK1181 fix).
if(MSVC AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
    target_link_directories(alphazero_chess PRIVATE "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    target_link_directories(import_ckpt PRIVATE "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
endif()

# Copy necessary runtime DLLs on Windows if needed
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET alphazero_chess POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:alphazero_chess>
    )
    add_custom_command(TARGET import_ckpt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:import_ckpt>
    )
endif()
