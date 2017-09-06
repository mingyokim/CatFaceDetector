cmake_minimum_required(VERSION 3.5)

project( detector )
set( EXECUTABLE_OUTPUT_PATH /home/min/CatProject/detector )

find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )
find_package( Boost 1.58.0 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /home/min/CatProject/detector/include )
include_directories( /usr/local/cuda/include )
include_directories( ${Boost_INCLUDE_DIR} )

add_definitions( -DGPU -DCUDNN )

set( SOURCE
  src/darknet.cpp
  src/detector.cpp
  src/main.cpp
  src/function.cpp
)

add_executable( detector ${SOURCE} )

find_library( DARKNET_LIB
  NAMES darknet
  PATHS /home/min/CatProject/detector/lib/
  NO_DEFAULT_PATH
)

find_library( BOOST_SYSTEM_LIB
  NAMES boost_system
  PATHS /usr/lib/x86_64-linux-gnu/
  NO_DEFAULT_PATH
)

find_library( BOOST_FILESYSTEM_LIB
  NAMES boost_filesystem
  PATHS /usr/lib/x86_64-linux-gnu/
  NO_DEFAULT_PATH
)

target_link_libraries( detector ${OpenCV_LIBS} ${DARKNET_LIB} )
target_link_libraries( detector cuda cudart cublas curand cudnn )
target_link_libraries( detector ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( detector ${BOOST_SYSTEM_LIB} )
target_link_libraries( detector ${BOOST_FILESYSTEM_LIB} )
#target_link_libraries( detector ${Boost_LIBRARIES} )
