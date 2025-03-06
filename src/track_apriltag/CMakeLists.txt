cmake_minimum_required(VERSION 3.8)
project(track_apriltag)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(cv_bridge REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(apriltag REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros)

add_executable(get_apriltag src/get_apriltag.cpp)
add_executable(check_opencv_version src/check_opencv_version.cpp)
add_executable(generate_aruco src/generate_aruco.cpp)
add_executable(get_3Dpose src/get_3Dpose.cpp)

ament_target_dependencies(generate_aruco
  rclcpp
  std_msgs
  geometry_msgs
  tf2_msgs
  #anafi_msg
)


ament_target_dependencies(check_opencv_version
  rclcpp
  std_msgs
  geometry_msgs
  tf2_msgs
  #anafi_msg
)

ament_target_dependencies(get_apriltag
  rclcpp
  std_msgs
  geometry_msgs
  tf2_msgs
  #anafi_msg
)

ament_target_dependencies(get_3Dpose
  rclcpp
  std_msgs
  geometry_msgs
  tf2_msgs
  sensor_msgs
  cv_bridge
  tf2_ros
  #anafi_msg
)

target_link_libraries(generate_aruco ${OpenCV_LIBRARIES})
target_link_libraries(check_opencv_version ${OpenCV_LIBRARIES})
target_link_libraries(get_3Dpose ${OpenCV_LIBRARIES})
#Install the executable
install (TARGETS
  get_3Dpose
  generate_aruco
  check_opencv_version
  get_apriltag
  DESTINATION lib/${PROJECT_NAME}
  )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
