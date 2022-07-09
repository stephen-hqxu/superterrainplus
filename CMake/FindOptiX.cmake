#[[
@brief Find Nvidia OptiX ray tracing API.
	The API structure has been changed significantly since version 7;
	SuperTerrain+ does not intend to complicate the engine by introducing backward compatibility to OptiX,
	therefore this module finder is designed for OptiX 7 (and potentially later versions).
------------------------------------------------------------------------------------------
Similar to most module finders, upon returning a number of variables are set. Those variables follow the standard CMake convention.
- `OptiX_FOUND` is set to TRUE if OptiX is found. Note that if the found version is incompatible it will be treated as not found.
- `OptiX_VERSION` is set to the version of found OptiX.
- `OptiX_INCLUDE_DIR` is set to where OptiX include directory is found.
- `OptiX_INSTALL_DIR` is set to where OptiX is installed on the system.
	This variable is also cached and user should set manually if it is not found.

Upon a successful attempt, an imported target OptiX::OptiX is provided.
]]

# find necessary headers
if(WIN32)
	# try to find it from default installation directory
	# find_path does not accept * symbol, use file to extract the information
	file(GLOB OPTIX_INSTALL_PREFIX "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")

	find_path(OptiX_INSTALL_DIR
	NAME include/optix.h
	PATHS ${OPTIX_INSTALL_PREFIX}
	DOC "Path to OptiX SDK installation location"
	)

	unset(OPTIX_INSTALL_PREFIX)
else()
	# try to get the installation directory from the environment
	set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX SDK installation location")
endif()
mark_as_advanced(OptiX_INSTALL_DIR)

# extract version information from the main header
if(OptiX_INSTALL_DIR)
	# set include directory
	set(OptiX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/include)

	# find the line containing raw version information
	file(STRINGS ${OptiX_INCLUDE_DIR}/optix.h OPTIX_VERSION_LINE
	REGEX "^#define OPTIX_VERSION [0-9]+"
	LIMIT_COUNT 1
	)
	# format the version line
	string(REGEX MATCH "[0-9]+" OPTIX_VERSION_LINE ${OPTIX_VERSION_LINE})
	# calculate version components
	math(EXPR OptiX_VERSION_MAJOR "${OPTIX_VERSION_LINE} / 10000")
	math(EXPR OptiX_VERSION_MINOR "(${OPTIX_VERSION_LINE} % 10000) / 100")
	math(EXPR OptiX_VERSION_MICRO "${OPTIX_VERSION_LINE} % 100")
	# concatenate the version numbers together
	set(OptiX_VERSION "${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_MICRO}")

	unset(OPTIX_VERSION_LINE)
endif()

# handling arguments as being called by find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
REQUIRED_VARS OptiX_INCLUDE_DIR
VERSION_VAR OptiX_VERSION
)

# create imported target
if(OptiX_FOUND AND NOT TARGET OptiX::OptiX)
	# OptiX is a header-only library, binaries reside along with CUDA
	add_library(OptiX::OptiX INTERFACE IMPORTED)
	# setup imported target properties
	set_target_properties(OptiX::OptiX PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES ${OptiX_INCLUDE_DIR}
	)
endif()