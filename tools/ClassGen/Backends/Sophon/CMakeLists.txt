
set(SOPHON_CLASS_GEN_BASE "" CACHE STRING "")

configure_file(SophonSpecificInstrsIF.h.in
    ${GLOW_BINARY_DIR}/glow/SophonSpecificInstrsIF.h)
configure_file(SophonSpecificNodesIF.h.in
    ${GLOW_BINARY_DIR}/glow/SophonSpecificNodesIF.h)

