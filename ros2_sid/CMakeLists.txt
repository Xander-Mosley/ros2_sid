cmake_minimum_required(VERSION 3.5)

#change name of package to this
project(ros2_sid)
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#https://docs.ros.org/en/foxy/How-To-Guides/Ament-CMake-Python-Documentation.html

# Find dependencies this will allow CPP and python
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Telem.msg"
#   "msg/CtlTraj.msg"

#   "srv/AddTwoInts.srv"
#   "srv/SetArmDisarm.srv"
#   "srv/SetGoal.srv"
#   "srv/SetUASType.srv"
#   "srv/SetTakeOff.srv"
#   # "srv/getGSInfo.srv"
#  )

# Include Cpp "include" directory
include_directories(include)
# Create Cpp executable
add_executable(main_pub src/main_pub.cpp src/minimal_publisher.cpp)
ament_target_dependencies(main_pub
    rclcpp
    std_msgs
)
# Install Cpp executables
install(TARGETS
  main_pub
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/pub_example.py
  scripts/sub_example.py
  scripts/goal_vis.py
  scripts/target_estimator.py
  scripts/guidance_publisher.py
  scripts/testingsignals_HINL.py
  scripts/testingsignals_SINL.py
  scripts/systemidentification.py
  # scripts/client_member_function.py
  # scripts/service_member_function.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()

##tests 
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(_pytest_tests
    tests/dummy_test.py
    # Add other test files here
  )
  foreach(_test_path ${_pytest_tests})
    get_filename_component(_test_name ${_test_path} NAME_WE)
    ament_add_pytest_test(${_test_name} ${_test_path}
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
      TIMEOUT 60
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  endforeach()
endif()

#launch
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)