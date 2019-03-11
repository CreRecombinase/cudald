cmake_minimum_required(VERSION 3.10)
find_package(PkgConfig REQUIRED)
enable_language(CXX)
set(BLA_VENDOR Intel10_64lp)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


SET(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cudald)
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)
find_package(HighFive REQUIRED)


pkg_check_modules(ZSTD REQUIRED libzstd)
find_path(ZSTD_INCLUDE_DIR zstd.h)

find_library(ZSTD_LIBRARY NAMES zstd)

if (ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
    set(ZSTD_FOUND TRUE)
    message(STATUS "Found ZSTD library: ${ZSTD_LIBRARY}")
endif ()


# message(STATUS "zstd_library:${ZSTD_LIBRARIES}")
# message(STATUS "zstd_library_dir:${ZSTD_LIBRARY_DIRS}")

add_executable(cudald ${cpu_source_files} src/main.cpp src/zstd_h5plugin.c src/zstd_h5plugin.h src/mkl_cov.hpp)
#find_library(PROTOBUF_LIBRARY zstd HINTS ${ZSTD_LIBRARY_DIRS})
add_library(mklcov src/mkl_cov.cpp)
target_link_libraries(mklcov ${MKL_LIBRARIES})
include_directories(${MKL_INCLUDE_DIRS})

add_library(cudacov src/cutest.cu src/cudacov.hpp)
target_link_libraries(cudacov -lcublas -lcurand)
target_include_directories(cudald PUBLIC ${xtensor_INCLUDE_DIRS})
target_include_directories(cudald PUBLIC ${ZSTD_INCLUDE_DIRS})
#target_link_libraries(cudald Z)

target_link_libraries(cudald HighFive)
target_link_libraries(cudald ${ZSTD_LIBRARY})
target_link_libraries(cudald xtensor)
target_link_libraries(cudald cudacov)
target_link_libraries(cudald mklcov)



message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
