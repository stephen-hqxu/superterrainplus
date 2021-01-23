#pragma once
#include "STPImageConverter.cuh"

using namespace SuperTerrainPlus::STPCompute;

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

__host__ STPImageConverter::STPImageConverter(uint2 mapSize) {
	//kernel launch parameters
	this->dimension = mapSize;
	this->numThreadperBlock = dim3(32, 32);
	this->numBlock = dim3(this->dimension.x / numThreadperBlock.x, this->dimension.y / numThreadperBlock.y);
}

__host__ STPImageConverter::~STPImageConverter() {

}

__host__ bool STPImageConverter::floatToshortCUDA(const float* const input, unsigned short* output, int channel) const {
	//the output, which should be _16 format
	const unsigned int num_channel = this->dimension.x * this->dimension.y * channel;
	bool no_error = true;
	//creating stream for async conversion
	cudaStream_t stream;
	no_error &= cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	//copy the host input to device
	float* input_d = nullptr;
	unsigned short* converted_d = nullptr;
	//allocating input and returning value
	no_error &= cudaSuccess == cudaMalloc(&input_d, sizeof(float) * num_channel);
	no_error &= cudaSuccess == cudaMalloc(&converted_d, sizeof(unsigned short) * num_channel);
	no_error &= cudaSuccess == cudaMemcpyAsync(input_d, input, sizeof(float) * num_channel, cudaMemcpyHostToDevice, stream);
	
	//call the kernel function
	STPKernelLauncher::floatToshortKERNEL<<<this->numBlock, this->numThreadperBlock, 0, stream>>>(input_d, converted_d, this->dimension, channel);

	//copy the value back to cpu
	no_error &= cudaSuccess == cudaMemcpyAsync(output, converted_d, sizeof(unsigned short) * num_channel, cudaMemcpyDeviceToHost, stream);

	//waiting until stream has finished
	no_error &= cudaSuccess == cudaStreamSynchronize(stream);

	//clear up and return
	no_error &= cudaSuccess == cudaFree(input_d);
	no_error &= cudaSuccess == cudaFree(converted_d);
	no_error &= cudaSuccess == cudaStreamDestroy(stream);
	return no_error;
}

__global__ void STPKernelLauncher::floatToshortKERNEL(const float* const input, unsigned short* output, uint2 dimension, int channel) {
	//current working pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		index = x + y * dimension.x;

	//loop through all channels and output
	for (int i = 0; i < channel; i++) {
		output[index * channel + i] = static_cast<unsigned short>(input[index * channel + i] * 65535u);
	}
}