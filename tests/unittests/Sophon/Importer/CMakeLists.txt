include_directories(${bmtap_SOURCE_DIR}/include)
include_directories(${BMNET_GLOW_INCLUDE_DIRS})

add_bmnet_glow_test(
    NAME caffeImporterTest
    SRCS caffeImporterTest.cpp
    LIBS bmnet-glow)

