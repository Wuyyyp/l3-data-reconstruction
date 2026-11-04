find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_OpenImageIO QUIET OpenImageIO)
endif()

find_path(OpenImageIO_INCLUDE_DIR
  NAMES OpenImageIO/imageio.h
  HINTS ${PC_OpenImageIO_INCLUDE_DIRS}
)
find_library(OpenImageIO_LIBRARY
  NAMES OpenImageIO
  HINTS ${PC_OpenImageIO_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenImageIO
  REQUIRED_VARS OpenImageIO_INCLUDE_DIR OpenImageIO_LIBRARY
  VERSION_VAR PC_OpenImageIO_VERSION
)

if(OpenImageIO_FOUND AND NOT TARGET OpenImageIO::OpenImageIO)
  add_library(OpenImageIO::OpenImageIO UNKNOWN IMPORTED)
  set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
    IMPORTED_LOCATION "${OpenImageIO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenImageIO_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(OpenImageIO_INCLUDE_DIR OpenImageIO_LIBRARY)
