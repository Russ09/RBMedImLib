cmake_minimum_required (VERSION 2.6)
project (RussLibTest)

include_directories("${PROJECT_SOURCE_DIR}/lib/nifti/niftilib")
include_directories("${PROJECT_SOURCE_DIR}/lib/nifti/znzlib")
include_directories("${PROJECT_SOURCE_DIR}/lib/include")
add_subdirectory(lib/src)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

Find_Package(PkgConfig)
pkg_check_modules(FFTWD fftw3 REQUIRED)
link_directories(${FFTWD_LIBRARY_DIRS})


set(EXTRA_LIBS ${EXTRA_LIBS} RBMedImLib fftw3)

add_executable(LibTest	applications/LibTest.cc)

target_link_libraries (LibTest  ${EXTRA_LIBS} ${FFTW_LIBRARIES})
