cmake_minimum_required(VERSION 3.10)
find_package(PkgConfig REQUIRED)
SET(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cudald)
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)
find_package(HighFive REQUIRED)

pkg_check_modules(ZSTD REQUIRED libzstd)
message(STATUS "zstd_library:${ZSTD_LIBRARIES}")

add_executable(cudald ${cpu_source_files} src/main.cpp src/zstd_h5plugin.c src/zstd_h5plugin.h)
add_library(cudacov src/cutest.cu src/cudacov.hpp)
target_link_libraries(cudacov -lcublas -lcurand)
target_include_directories(cudald PUBLIC ${xtensor_INCLUDE_DIRS})
target_include_directories(cudald PUBLIC ${ZSTD_INCLUDE_DIRS})

target_link_libraries(cudald HighFive)
target_link_libraries(cudald PUBLIC ${ZSTD_LIBRARIES})

target_link_libraries(cudald xtensor)
target_link_libraries(cudald cudacov)


message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
