cmake_minimum_required(VERSION 3.8)
project(htmslam)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)  # <-- IMPORTANTE
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(topological_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(message_filters REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Irrlicht REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(roaring REQUIRED)

# Find Python and pybind11 for visual feature extractor
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

find_package(nlohmann_json REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

find_package(ament_index_cpp REQUIRED)

# CRITICAL: Enable optimization flags for Eigen performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
add_definitions(-DEIGEN_NO_DEBUG)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# uncomment if you don't have irrlicht installed
add_definitions("-DHAVE_IRRLICHT")


rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ImportMap.srv"
  "srv/ListMaps.srv"
  "srv/SetGoal.srv"  # <-- NOVO
  DEPENDENCIES builtin_interfaces
)

# Targets

# visual_feature_extractor_node executable
add_executable(visual_feature_extractor_node 
  src/visual_feature_extractor/visual_feature_extractor_node.cpp)
target_include_directories(visual_feature_extractor_node PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
)
target_link_libraries(visual_feature_extractor_node 
  pybind11::embed
  Python3::Python
)
ament_target_dependencies(visual_feature_extractor_node
  rclcpp
  std_msgs
  sensor_msgs
  topological_msgs
)

# binary_projector_node executable
add_executable(binary_projector_node 
  src/dim_reduction_and_binarization/binary_projector_node.cpp)
target_include_directories(binary_projector_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(binary_projector_node
  rclcpp
  topological_msgs
)

# neocortex_node executable
add_executable(neocortex_node src/neocortex/neocortex_node.cpp)
target_include_directories(neocortex_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neocortex
)
target_link_libraries(neocortex_node 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neocortex/htm/libhtm_core.a
  pthread
)
ament_target_dependencies(neocortex_node
  rclcpp
  topological_msgs
)

# pcd_htm_node executable (PointCloudRecognition: encoder SDR + HTM context)
add_executable(pcd_htm_node
  src/PointCloudRecognition/src/pcd_htm_node.cpp)
target_include_directories(pcd_htm_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neocortex
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(pcd_htm_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neocortex/htm/libhtm_core.a
  nlohmann_json::nlohmann_json
  pthread
)
ament_target_dependencies(pcd_htm_node
  rclcpp
  sensor_msgs
  topological_msgs
  ament_index_cpp
)

# place_context_cells_node executable (RatSLAM-style local view cells)
add_executable(place_context_cells_node
  src/place_context_cells/place_context_cells_node.cpp)
target_include_directories(place_context_cells_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PointCloudRecognition/src
)
ament_target_dependencies(place_context_cells_node
  rclcpp
  sensor_msgs
  message_filters
  topological_msgs
)

# spatial_view_cells_node executable
add_executable(spatial_view_cells_node 
  src/spatial_view_cells/spatial_view_cells_node.cpp)
target_include_directories(spatial_view_cells_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/spatial_view_cells
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager/include
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(spatial_view_cells_node 
  roaring::roaring
  ${JSONCPP_LIBRARIES}
)
ament_target_dependencies(spatial_view_cells_node
  rclcpp
  std_msgs
  std_srvs
  topological_msgs
  ament_index_cpp
)

# pose_cells_node executable
add_executable(pose_cells_node  
  src/pose_cells/pose_cells_node.cpp
  src/pose_cells/posecell_network.cpp)

target_include_directories(pose_cells_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pose_cells
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager/include 
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(pose_cells_node
  ${IRRLICHT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${JSONCPP_LIBRARIES}
)
ament_target_dependencies(pose_cells_node
  rclcpp
  nav_msgs
  std_srvs
  topological_msgs
  ament_index_cpp
)


add_executable(experience_map_node 
  src/experience_map/experience_map_node.cpp
  src/experience_map/experience_map.cpp
  src/experience_map/map_manager.cpp
)

target_include_directories(experience_map_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/experience_map
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager/include  
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pose_cells
  ${CMAKE_CURRENT_SOURCE_DIR}/src/spatial_view_cells
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp 
  ${JSONCPP_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

target_link_libraries(experience_map_node
  ${Boost_LIBRARIES}
  ${IRRLICHT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${PROJECT_NAME}__rosidl_typesupport_cpp
  roaring::roaring
)

ament_target_dependencies(experience_map_node
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  topological_msgs
  std_srvs
  yaml-cpp
  rosidl_default_runtime
  ament_index_cpp
)

# add_dependencies(experience_map_node ${PROJECT_NAME}_rosidl_generator_cpp)

# visual_odometry_node executable
add_executable(visual_odometry_node 
  src/visual_odometry/visual_odometry_node.cpp
  src/visual_odometry/visual_odometry.cpp)
target_include_directories(visual_odometry_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/visual_odometry
)
target_link_libraries(visual_odometry_node
  ${OpenCV_LIBRARIES}
)
ament_target_dependencies(visual_odometry_node
  rclcpp
  sensor_msgs
  nav_msgs
  topological_msgs
)

# mode_manager_node executable
add_executable(mode_manager_node 
  src/mode_manager/mode_manager_node.cpp
)

# Adiciona os includes do mode_manager
target_include_directories(mode_manager_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mode_manager/include
)

ament_target_dependencies(mode_manager_node
  rclcpp
  std_msgs
  std_srvs
)


install(TARGETS 
  visual_feature_extractor_node
  binary_projector_node
  neocortex_node
  pcd_htm_node
  place_context_cells_node
  spatial_view_cells_node 
  pose_cells_node 
  experience_map_node 
  visual_odometry_node
  mode_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install service interface files
install(DIRECTORY srv/
  DESTINATION share/${PROJECT_NAME}/srv
  FILES_MATCHING PATTERN "*.srv"
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY src/media/
  DESTINATION share/${PROJECT_NAME}/media
)

# Install Python module for visual feature extractor
install(DIRECTORY src/visual_feature_extractor/
  DESTINATION lib/${PROJECT_NAME}/visual_feature_extractor
  FILES_MATCHING PATTERN "*.py"
)

# Install random matrix for binary projector
install(DIRECTORY src/dim_reduction_and_binarization/random_matrix/
  DESTINATION share/${PROJECT_NAME}/random_matrix
  FILES_MATCHING PATTERN "*.bin"
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# Place encoder configuration used by pcd_htm_node
install(FILES src/PointCloudRecognition/src/encoder_config.json
  DESTINATION share/${PROJECT_NAME}/config
)

# Install headers
install(DIRECTORY src/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h"
)

# Instalar headers do mode_manager
install(DIRECTORY src/mode_manager/include/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()