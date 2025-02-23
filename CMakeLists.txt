cmake_minimum_required(VERSION 2.8)
#onlyliucat

project(MultiChessboardExtraction)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -fPIC  -lm -Wl,--no-as-needed -DLINUX -g -Wall \
#-std=c++11 -Wunused-variable -Wwrite-strings -Wsign-compare -fpermissive ")
#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OpenCV_LIBRARIES})
MESSAGE(STATUS "The Opencv's include directory is:" ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "The Opencv's OpenCV_LIBS is:" ${OpenCV_LIBS})


INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(src)
AUX_SOURCE_DIRECTORY(src gSOURCE_FILES_)


add_executable(${PROJECT_NAME} "src/main.cpp"
        ${gSOURCE_FILES_}
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
