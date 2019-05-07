cmake_minimum_required(VERSION 2.8.3)
project(refusion)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++11;-arch=sm_50;-O3)

catkin_package(
  INCLUDE_DIRS src
  LIBRARIES refusion
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(
src
${catkin_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
)

add_library(eigen_wrapper src/tracker/eigen_wrapper.cpp)
cuda_add_library(refusion STATIC
  src/utils/rgbd_image.cu
  src/marching_cubes/mesh.cu
  src/marching_cubes/mesh_extractor.cu
  src/tsdfvh/heap.cu
  src/tsdfvh/hash_table.cu 
  src/tsdfvh/tsdf_volume.cu
  src/tracker/tracker.cu)

target_link_libraries(refusion eigen_wrapper ${CUDA_LIBRARIES})

add_executable(refusion_example src/example/example.cpp src/example/fr_parser.cpp)

target_link_libraries(refusion_example
  refusion
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

# Generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile @ONLY)
  add_custom_target(doc ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)
