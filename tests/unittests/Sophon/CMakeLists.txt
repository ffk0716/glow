

include_directories(${Glow_SOURCE_DIR}/lib)

function(add_bmnet_glow_test)
    cmake_parse_arguments(ARG "" "NAME" "SRCS;LIBS;ARGS" ${ARGN})
    add_executable(${ARG_NAME} ${ARG_SRCS})
    target_link_libraries(${ARG_NAME} PRIVATE
        ${ARG_LIBS}
        Sophon
        gtest
        testMain)
    add_test(
        NAME sophon_${ARG_NAME}
        COMMAND ${ARG_NAME} ${ARG_ARGS})
endfunction()

add_subdirectory(Backends)
# add_subdirectory(Importer)
