file(GLOB_RECURSE cu_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(cufile IN LISTS cu_files)
#    # Get file name without directory
     get_filename_component(mainname ${cufile} NAME_WE)
     cuda_add_executable(${mainname} ${cufile})
#     add_executable(cuda ${cudile})
endforeach()