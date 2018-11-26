cmake_minimum_required(VERSION 3.12)
project(oclbhgs C)

set(CMAKE_C_STANDARD 99)

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(HEADERS
        ./sources/headers/galaxy.h
        ./sources/headers/cell.h
        ./sources/headers/body.h
        ./sources/headers/mass.h
        ./sources/headers/com.h
        ./sources/headers/position.h
        ./sources/headers/ocl.h
        ./sources/headers/ogl.h
        ./sources/headers/keys.h

        ./sources/kernel_headers/galaxy.h
        ./sources/kernel_headers/cell.h
        ./sources/kernel_headers/body.h
        ./sources/kernel_headers/mass.h
        ./sources/kernel_headers/com.h
        ./sources/kernel_headers/position.h
        ./sources/kernel_headers/ocl.h
        )

set(SOURCES
        ./sources/logics/galaxy.c
        ./sources/logics/body.c
        ./sources/logics/cell.c
        ./sources/logics/ocl.c
        ./sources/logics/ogl.c

        )

add_compile_definitions(OPENCL_HOST=true)

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-O3 -Wall -Wextra -Wno-deprecated")

add_executable(oclbhgs main.c ${HEADERS} ${SOURCES})

if(APPLE)
    target_link_libraries(oclbhgs ${OpenCL_LIBRARY} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY} ${GLUT_LIBRARY} pthread m)
else()
    target_link_libraries(oclbhgs ${OpenCL_LIBRARY} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY} ${GLUT_LIBRARY} GLU pthread m)
endif()

target_include_directories(oclbhgs PRIVATE ${OpenCL_INCLUDE_DIRS} ./cmake-build-debug ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${GLUT_INCLUDE_DIRS})
