cmake_minimum_required(VERSION 3.5)
project(anafi_msg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Find Fast-RTPS dependencies
find_package(rosidl_typesupport_fastrtps_c REQUIRED)
find_package(rosidl_typesupport_fastrtps_cpp REQUIRED)

# Generate ROS interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/Position.msg"
    "msg/Speed.msg"
    "msg/CurrentState.msg"
    "msg/Output.msg"
    "msg/CollectCurrentState.msg"
    "msg/Matrices.msg"
    "msg/MovingReference.msg"
    "msg/JackalState.msg"
    "msg/BoxData.msg"
    "msg/Bbox.msg"
    "msg/KpData.msg"
    "msg/DetectronBox.msg"
    "msg/KeyPoints.msg"
    "msg/ControlData.msg"
    "msg/PnPData.msg"
    "msg/DroneSize.msg"
    "msg/ImageKeyPoints.msg"
    "msg/DroneVehiclePos.msg"
    "msg/PnPDataYolo.msg"
    "msg/KpYolo.msg"
    "msg/FrameYolo.msg"
    "msg/PlotData.msg"

  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

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
