#pragma once
#ifndef _STP_IMAGE_CONVERTER_CUH_
#define _STP_IMAGE_CONVERTER_CUH_

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {

		/**
		 * @brief STPImageConverter is a class containing tools to convert image color format.
		 * Currently it can only convert image from high quality to low quality.
		 * It only affect the color format of the texture, e.g., how many bits each pixel uses.
		*/
		class STPImageConverter {
		private:

			//The size of the image, since we only uses it for our gnerated maps which are all defined in the same size, such that we don't need to pass the variable everytime.
			uint2 dimension;
			dim3 numThreadperBlock, numBlock;

		public:

			typedef unsigned int STPfloat16;

			/**
			 * @brief Init the image converter with a fixed image dimension
			 * @param mapSize The size of all maps that needs to be converted
			*/
			__host__ STPImageConverter(uint2);

			__host__ ~STPImageConverter();

			/**
			 * @brief Convert _32F format to _16
			 * @param input The input image, each color channel occupies 32 bit (float)
			 * @param output The output image, each color channel occupies 16 bit (unsigne short int).
			 * @param channel How many channel in the texture, the input and output channel will have the same number of channel
			 * @return True if conversion was successful without errors
			*/
			__host__ bool floatToshortCUDA(const float* const, unsigned short*, int);

			/**
			 * @brief Convert _32F format to _16F
			 * @param input The input image, each color channel occupies 32 bit (float)
			 * @param outptu The output image, each color channel occupies 16 bit half float (unsigned short int will be used to interpret IEEE-754 half float format).
			 * @param channel How many channel in the texture, the input and output channel will have the same number of channel
			 * @return True if conversion was successful without errors
			*/
			__host__ bool floatTohalfCUDA(const float* const, STPfloat16*, int);
			
		};

		/**
		 * @brief Kernel launch and util functions
		*/
		namespace STPKernelLauncher {

			/**
			 * @brief Performing the conversion from _32F to _16
			 * @param input The input image, each color channel occupies 32 bit (float)
			 * @param output The output image, each color channel occupies 16 bit (unsigne short int)
			 * @param dimension The size of the map
			 * @param channel How many channel in the texture, the input and output channel will have the same number of channel
			*/
			__global__ void floatToshortKERNEL(const float* const, unsigned short*, uint2, int);

			/**
			 * @brief Convert _32F format to _16F
			 * @param input The input image, each color channel occupies 32 bit (float)
			 * @param output The output image, each color channel occupies 16 bit half float (unsigned short int will be used to interpret IEEE-754 half float format)
			 * @param dimension The size of the map
			 * @param channel How many channel in the texture, the input and output channel will have the same number of channel
			*/
			__global__ void floatTohalfKERNEL(const float* const, STPImageConverter::STPfloat16*, int2, int);
		}

	}
}
#endif//_STP_IMAGE_CONVERTER_CUH_