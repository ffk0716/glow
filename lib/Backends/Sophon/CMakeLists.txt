

# prepare Sophon external lib
# get more info from https://sophon-edge.gitbook.io/project/getting-started/bmnnsdk-installation
include(ExternalProject)
set(Sophon_ext_lib ${CMAKE_CURRENT_BINARY_DIR}/sophon_sdk/src/sophon_sdk/bmtap2-bm1880-usb_1.0.2)
ExternalProject_Add(sophon_sdk
    PREFIX sophon_sdk
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
    ${Sophon_ext_lib}/lib/cmodel/libbmkernel-static.a
    ${Sophon_ext_lib}/lib/cmodel/libbmodel.so
    ${Sophon_ext_lib}/lib/cmodel/libbmruntime.so
    GIT_REPOSITORY https://github.com/ffk0716/bm1880-bmnnsdk-usb.git
    GIT_TAG bm1880-usb_1.0.2.2-hotfix)

# Sophon Backend
find_package(Protobuf REQUIRED)
include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
include_directories(SYSTEM ${Sophon_ext_lib}/bmnet_sdk/include)
include_directories(SYSTEM ${Sophon_ext_lib}/bmnet_sdk/install/include)
include_directories(${Glow_SOURCE_DIR}/lib)

add_library(Sophon "")

target_link_libraries(Sophon PRIVATE
    Backends
    CodeGen
    Importer
    Quantization
    ${Sophon_ext_lib}/bmnet_sdk/install/lib/libbmkernel-static.a
    ${Sophon_ext_lib}/bmnet_sdk/install/lib/libbmodel.so
    ${Sophon_ext_lib}/bmnet_sdk/install/lib/libbmruntime.so)

set(Sophon_base ${Glow_SOURCE_DIR}/lib/Backends)
function(add_libbmnet_glow)
    file(RELATIVE_PATH target_name ${Sophon_base} ${CMAKE_CURRENT_SOURCE_DIR})
    string(REPLACE "/" "_" target_name ${target_name})
    string(REPLACE "-" "_" target_name ${target_name})
    set(target_name ${target_name}_obj)
    add_library(${target_name} OBJECT ${ARGN})
    add_dependencies(${target_name} sophon_sdk)
    target_sources(Sophon PRIVATE $<TARGET_OBJECTS:${target_name}>)
endfunction()


add_libbmnet_glow(
    Bundle.cpp
    CommandLine.cpp
    GlowLIRVisitor.cpp
    SophonBackend.cpp
    SophonFunction.cpp
    SophonQuantizer.cpp)

add_subdirectory(BM188x)
add_subdirectory(Utility)
