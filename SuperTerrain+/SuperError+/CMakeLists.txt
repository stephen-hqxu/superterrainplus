add_library(${STP_ERROR_ENGINE} SHARED
STPDeviceErrorHandler.cpp
)

set_target_properties(${STP_ERROR_ENGINE} PROPERTIES DEFINE_SYMBOL STPERRORPLUS_EXPORTS)

# link CUDA library
target_link_libraries(${STP_ERROR_ENGINE}
PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
)