cmake_minimum_required (VERSION 3.31)

project (cndo/2_molecular_properties LANGUAGES CXX)

# Compiler requirements
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find JSON Libraries
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(chemfiles REQUIRED)

add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    Eigen3::Eigen
    armadillo
    nlohmann_json::nlohmann_json
    HDF5::HDF5
    chemfiles
)

target_compile_definitions(common INTERFACE 
    ARMA_USE_HDF5
    H5_USE_110_API
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|x86_64|AMD64")
    add_compile_options(-Wa,--gsframe=no)
endif()

link_libraries(common)

# If Needed Make Modifications below this line 

# Reusable CNDO/2 computational library
add_library(cndo2 STATIC 
    src/cndo2_helpers.cpp
    src/molecular_properties.cpp
)

target_include_directories(cndo2 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src 
)

# Existing command-line application 
add_executable(hw_5_2
    src/hw_5_2.cpp 
)

target_link_libraries(hw_5_2 PRIVATE cndo2)

# Enable CTest for automated testing
enable_testing()

# Unit test for atomic partial charges
add_executable(test_atomic_charges
    tests/test_atomic_charges.cpp
)

# Link the test executable to our CNDO/2 Library
target_link_libraries(test_atomic_charges PRIVATE cndo2)

# Register the test with CTest
add_test(
    NAME AtomicCharges 
    COMMAND test_atomic_charges
)

# Integration test using actual CNDO/2 SCF results 
add_executable(test_scf_atomic_charges 
    tests/test_scf_atomic_charges.cpp)

target_link_libraries(test_scf_atomic_charges PRIVATE cndo2)

add_test(
    NAME SCFAtomicCharges
    COMMAND test_scf_atomic_charges
)

# Run from the project root so the solver can find basis/file 
set_tests_properties(SCFAtomicCharges PROPERTIES 
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Unit tests for charge-based molecular dipole moments
add_executable(test_dipole_moment 
    tests/test_dipole_moment.cpp
)

target_link_libraries(test_dipole_moment PRIVATE cndo2)

add_test(
    NAME DipoleMoment
    COMMAND test_dipole_moment
)