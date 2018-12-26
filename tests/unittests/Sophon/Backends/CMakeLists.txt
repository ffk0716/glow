include_directories(${bmtap_SOURCE_DIR}/include)
include_directories(${BMNET_GLOW_INCLUDE_DIRS})

########################
#  Target
set(Sophon_CHIP_LIST BM1680 BM1682 BM1880 BM1882)
string(REPLACE ";" ", " Sophon_CHIP_LIST_HELP "${Sophon_CHIP_LIST}")
set(Sophon_CHIP BM1880 CACHE STRING "chip version: ${Sophon_CHIP_LIST_HELP}")
set_property(CACHE Sophon_CHIP PROPERTY STRINGS ${Sophon_CHIP_LIST})
set(Sophon_${Sophon_CHIP}_ENABLE ON)

if (Sophon_BM1880_ENABLE)
    add_glow_sophon_test(
        NAME BM1880CodeGenTest
        SRCS BM1880CodeGenTest.cpp
        ARGS -enable-layer-group=false)
    add_glow_sophon_test(
        NAME BM1880Expand
        SRCS BM1880Expand.cpp)
    add_glow_sophon_test(
        NAME BM1880ExpandCodeGenTest
        SRCS BM1880ExpandCodeGenTest.cpp)
    add_glow_sophon_test(
        NAME BM1880GetLMemSizeTest
        SRCS BM1880GetLMemSizeTest.cpp)
    add_glow_sophon_test(
        NAME BM1880MemoryAllocTest
        SRCS BM1880MemoryAllocTest.cpp)
    add_glow_sophon_test(
        NAME BM1880DeleteQuantizeNodeTest
        SRCS BM1880DeleteQuantizeNodeTest.cpp)
    add_glow_sophon_test(
        NAME BM1880HIRSliceTest
        SRCS BM1880HIRSliceTest.cpp)
endif()

add_glow_sophon_test(
    NAME LoadStoreTest
    SRCS LoadStoreTest.cpp)
