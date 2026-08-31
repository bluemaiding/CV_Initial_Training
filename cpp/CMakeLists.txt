cmake_minimum_required(VERSION 3.10)
project(YoloOnnxInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 请把下面路径设置为你本地解压/安装的目录
# ONNXRUNTIME_DIR 应指向 onnxruntime zip 解压后的根目录（包含 include/ 和 lib/ 子目录）
set(ONNXRUNTIME_DIR "C:/Users/bluem/Downloads/onnxruntime-win-x64-1.28.1/onnxruntime-win-x64-1.28.1")
# OpenCV_DIR 应指向包含 OpenCVConfig.cmake 的目录，例如 C:/opencv/build 或 C:/opencv/build/x64/vc16
set(OpenCV_DIR "C:/opencv/build")

# 找到 OpenCV（如果 OpenCV_DIR 设置正确，CMake 会找到）
find_package(OpenCV REQUIRED)

# 如果 onnxruntime 没有 CMake config 文件或你没有通过 vcpkg 安装，则手动添加 include/lib
if(EXISTS "${ONNXRUNTIME_DIR}/include" AND EXISTS "${ONNXRUNTIME_DIR}/lib")
    message(STATUS "Using ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
    set(ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_DIR}/lib")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_PATH}/onnxruntime.lib")
else()
    message(FATAL_ERROR "ONNXRUNTIME_DIR not configured correctly. Make sure it contains include/ and lib/ subfolders.")
endif()

add_executable(inference inference.cpp)
target_compile_features(inference PRIVATE cxx_std_17)

# 包含目录
target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})

# 链接库（使用 OpenCV 提供的变量以及 onnxruntime.lib）
target_link_libraries(inference PRIVATE ${OpenCV_LIBS} "${ONNXRUNTIME_LIB}")

# 将 onnxruntime 的 DLL 在构建后复制到可执行文件所在目录，避免运行时找不到 DLL
add_custom_command(TARGET inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_PATH}/onnxruntime.dll"
        $<TARGET_FILE_DIR:inference>
)