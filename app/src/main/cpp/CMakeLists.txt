cmake_minimum_required(VERSION 3.22.1)
project("depthlane")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("/Users/lauraperez/Downloads/OpenCV-android-sdk/sdk/native/jni/include")

add_library(
    native-lib
    SHARED
    native-lib.cpp
)

add_library(opencv_java SHARED IMPORTED)
set_target_properties(opencv_java PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java4.so"
)

target_link_libraries(
    native-lib
    opencv_java
    android
    log
)

target_link_options(
    native-lib
    PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
)
