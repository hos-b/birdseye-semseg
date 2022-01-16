cmake_minimum_required (VERSION 3.12)
# cmake_policy(SET CMP0048 NEW)
project(mass_data_collector)

# cmake flags ---------------------------------------------------------------
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_FIND_DEBUG_MODE ON)

# required for stuff and things ---------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3 REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
find_package(nanoflann)
find_package(OpenMP REQUIRED)
# CARLA related stuff -------------------------------------------------------
if (NOT DEFINED ENV{CARLA_910_CXX_API})
	message(FATAL_ERROR "environment variable CARLA_910_CXX_API is not defined.")
endif ()
FILE(GLOB CARLA_LIBRARIES $ENV{CARLA_910_CXX_API}/lib/*.a)
# catkin packages -----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	# cv_bridge
	# image_transport
	# sensor_msgs
	# freicar_map
)
# HDF5 ----------------------------------------------------------------------
set(LIBHDF5_ROOT /usr/local/HDF_Group/HDF5/1.12.0)
list (APPEND CMAKE_PREFIX_PATH ${LIBHDF5_ROOT})
find_package (HDF5 NAMES "hdf5" COMPONENTS C CXX HL HLCXX static) # NAMES hdf5 COMPONENTS C ${SEARCH_TYPE}
if (HDF5_FOUND)
  if (HDF5_static_C_FOUND)
    set (HDF5_LIBS ${HDF5_LIBS} ${HDF5_C_STATIC_LIBRARY})
  endif ()
  if (HDF5_static_CXX_FOUND)
    set (HDF5_LIBS ${HDF5_LIBS} ${HDF5_CXX_STATIC_LIBRARY})
  endif ()
else ()
  message (FATAL_ERROR " HDF5 is Required")
endif ()

# more catkin stuff ---------------------------------------------------------
catkin_package(CATKIN_DEPENDS 
	roscpp
	roslib
	# cv_bridge
	# image_transport
	# sensor_msgs
	# freicar_map
)

# include dirs --------------------------------------------------------------
include_directories(
	"/usr/lib/gcc/x86_64-linux-gnu/9/include;"
	${PROJECT_SOURCE_DIR}/include
	${HDF5_INCLUDE_DIR}
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	${OPEN3D_INCLUDE_DIRS}
	$ENV{CARLA_910_CXX_API}/include
	$ENV{CARLA_910_CXX_API}/include/system
)
# HDF5 lib for the dataset --------------------------------------------------
add_library(hdf5_dataset STATIC 
	${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_dataset.cpp
)
target_link_libraries(hdf5_dataset ${HDF5_LIBS})
set_target_properties(hdf5_dataset PROPERTIES CXX_STANDARD 17)

# HDF5 api test -------------------------------------------------------------
add_executable(hdf5_test ${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_test.cpp)
target_link_libraries(hdf5_test hdf5_dataset)

# data collection executable ------------------------------------------------
add_executable(${PROJECT_NAME}
	src/main.cpp
	src/geometry/camera_geometry.cpp
	src/geometry/semantic_cloud.cpp
	src/mass_agent/mass_agent.cpp
	src/mass_agent/sensors.cpp
	src/config/geom_config.cpp
	src/config/agent_config.cpp
)
target_link_libraries(${PROJECT_NAME}
	$ENV{CARLA_910_CXX_API}/lib/libDetourCrowd.a
	$ENV{CARLA_910_CXX_API}/lib/libDetour.a
	$ENV{CARLA_910_CXX_API}/lib/librpc.a
	${catkin_LIBRARIES}
	${CARLA_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
	nanoflann::nanoflann
	OpenMP::OpenMP_CXX
	Eigen3::Eigen
	hdf5_dataset
	yaml-cpp
)

# runtime collection executable ------------------------------------------------
add_executable(runtime
	src/runtime.cpp
	src/geometry/camera_geometry.cpp
	src/geometry/semantic_cloud.cpp
	src/mass_agent/mass_agent.cpp
	src/mass_agent/sensors.cpp
	src/config/geom_config.cpp
	src/config/agent_config.cpp
)
target_link_libraries(runtime
	$ENV{CARLA_910_CXX_API}/lib/libDetourCrowd.a
	$ENV{CARLA_910_CXX_API}/lib/libDetour.a
	$ENV{CARLA_910_CXX_API}/lib/librpc.a
	${catkin_LIBRARIES}
	${CARLA_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
	nanoflann::nanoflann
	OpenMP::OpenMP_CXX
	Eigen3::Eigen
	hdf5_dataset
	yaml-cpp
)

# sample viz executable -----------------------------------------------------
add_executable(testviz
	src/visualizer/mviz.cpp
	src/geometry/camera_geometry.cpp
	src/geometry/semantic_cloud.cpp
	src/mass_agent/mass_agent.cpp
	src/mass_agent/sensors.cpp
	src/config/geom_config.cpp
	src/config/agent_config.cpp
)
target_link_libraries(testviz
	$ENV{CARLA_910_CXX_API}/lib/libDetourCrowd.a
	$ENV{CARLA_910_CXX_API}/lib/libDetour.a
	$ENV{CARLA_910_CXX_API}/lib/librpc.a
	${catkin_LIBRARIES}
	${CARLA_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
	nanoflann::nanoflann
	OpenMP::OpenMP_CXX
	Eigen3::Eigen
	yaml-cpp
)