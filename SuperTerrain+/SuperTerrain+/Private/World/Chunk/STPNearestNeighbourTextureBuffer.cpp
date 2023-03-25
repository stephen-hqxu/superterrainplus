#include <SuperTerrain+/World/Chunk/STPNearestNeighbourTextureBuffer.h>
//Chunk
#include <SuperTerrain+/World/Chunk/STPChunk.h>

//CUDA
#include <cuda_runtime.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

#include <cassert>

using namespace SuperTerrainPlus;
using STPDiversity::Sample;

using glm::uvec2;

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::STPMergedBuffer(
	const STPNearestNeighbourTextureBuffer& nn_texture_buffer, const STPMemoryLocation location) : Main(nn_texture_buffer), Location(location) {
	const unsigned int total_pixel = this->Main.calcNeighbourPixel();
	const auto [mem_pool, stream] = this->Main.DeviceMemInfo;

	//allocate host memory, we need host memory regardless of memory location
	this->HostMem = STPSmartDeviceMemory::makeHost<MutableT[]>(total_pixel);
	if constexpr (MM != STPNearestNeighbourTextureBufferMemoryMode::WriteOnly) {
		//need to copy from each neighbour chunk to a merged buffer, in read only or read/write mode
		this->copyNeighbourTexture<true>();
	}
	
	if (this->Location == STPMemoryLocation::DeviceMemory) {
		//need to allocate device memory
		this->DeviceMem = STPSmartDeviceMemory::makeStreamedDevice<MutableT[]>(mem_pool, stream, total_pixel);
		if constexpr (MM != STPNearestNeighbourTextureBufferMemoryMode::WriteOnly) {
			//same as what we are doing to host memory
			STP_CHECK_CUDA(cudaMemcpyAsync(this->DeviceMem.get(), this->HostMem.get(), total_pixel * sizeof(T),
				cudaMemcpyHostToDevice, stream));
		}
	}
}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::~STPMergedBuffer() {
	if (!this->HostMem) {
		//if host memory is not allocated, meaning this merged buffer is empty, nothing to do...
		assert(!this->DeviceMem);
		return;
	}
	const auto [mem_pool, stream] = this->Main.DeviceMemInfo;
	const size_t total_buffer_size = this->Main.calcNeighbourPixel() * sizeof(T);
	
	if constexpr (MM != STPNearestNeighbourTextureBufferMemoryMode::ReadOnly) {
		//we need to copy the large buffer back to each chunk when operating on a non-read-only mode
		if (this->Location == STPMemoryLocation::DeviceMemory) {
			//copy device memory to pinned memory we have allocated previously
			STP_CHECK_CUDA(cudaMemcpyAsync(this->HostMem.get(), this->DeviceMem.get(), total_buffer_size,
				cudaMemcpyDeviceToHost, stream));
		}

		//un-merge buffer back to each neighbour chunk
		this->copyNeighbourTexture<false>();
	}
	//we don't need to copy the texture back to the original buffer if it's read only

	//make sure all works are done before pinned memory is automatically destroyed
	STP_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
template<bool Pack>
void STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::copyNeighbourTexture() {
	std::conditional_t<Pack, MutableT* const, const MutableT* const> host_accumulator = this->HostMem.get();

	TextureType* const* const neighbour_texture = this->Main.NeighbourTexture;
	const STPNearestNeighbourInformation& info = this->Main.NeighbourInfo;

	const uvec2 chunk_nn = info.ChunkNearestNeighbour;
	const unsigned int neighbour_count = chunk_nn.x * chunk_nn.y;

	const cudaStream_t stream = this->Main.DeviceMemInfo.second;
	//make a initial copy from the original buffer if it's not write only
	//combine texture from each chunk to a large buffer
	if (neighbour_count == 1u) {
		const size_t pixel_size = this->Main.calcChunkPixel() * sizeof(T);

		//no nearest neighbour logic, a simple linear memory copy suffices
		if constexpr (Pack) {
			STP_CHECK_CUDA(cudaMemcpyAsync(host_accumulator, *neighbour_texture, pixel_size, cudaMemcpyHostToHost, stream));
		} else {
			STP_CHECK_CUDA(cudaMemcpyAsync(*neighbour_texture, host_accumulator, pixel_size, cudaMemcpyHostToHost, stream));
		}
		return;
	}

	const size_t map_row_size = info.MapSize.x * sizeof(T),
		total_row_size = map_row_size * chunk_nn.x;
	//copy with free-slip logic using 2D copy
	for (unsigned int i = 0u; i < neighbour_count; i++) {
		//the local coordinate of the current chunk
		const uvec2 local_offset = STPChunk::calcLocalChunkCoordinate(i, chunk_nn),
			//the pixel coordinate at the top-left corner of the current chunk
			local_pixel_offset = info.MapSize * local_offset;
		//convert that to linear offset
		const unsigned int offset = local_pixel_offset.x + local_pixel_offset.y * info.TotalMapSize.x;

		if constexpr (Pack) {
			STP_CHECK_CUDA(cudaMemcpy2DAsync(host_accumulator + offset, total_row_size, neighbour_texture[i],
				map_row_size, map_row_size, info.MapSize.y, cudaMemcpyHostToHost, stream));
		} else {
			STP_CHECK_CUDA(cudaMemcpy2DAsync(neighbour_texture[i], map_row_size, host_accumulator + offset,
				total_row_size, map_row_size, info.MapSize.y, cudaMemcpyHostToHost, stream));
		}
	}
}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
typename STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::MutableT* STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::getHost() const noexcept {
	return this->HostMem.get();
}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
typename STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::MutableT* STPNearestNeighbourTextureBuffer<T, MM>::STPMergedBuffer::getDevice() const noexcept {
	return this->DeviceMem.get();
}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
STPNearestNeighbourTextureBuffer<T, MM>::STPNearestNeighbourTextureBuffer(TextureType* const* const texture,
	const STPNearestNeighbourInformation& nn_info,
	const STPDeviceMemoryOperator& texture_device_mem_alloc) : NeighbourInfo(nn_info), DeviceMemInfo(texture_device_mem_alloc),
	NeighbourTexture(texture) {

}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
inline unsigned int STPNearestNeighbourTextureBuffer<T, MM>::calcChunkPixel() const noexcept {
	const uvec2 dimension = this->NeighbourInfo.MapSize;
	return dimension.x * dimension.y;
}

template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
inline unsigned int STPNearestNeighbourTextureBuffer<T, MM>::calcNeighbourPixel() const noexcept {
	const uvec2 total = this->NeighbourInfo.TotalMapSize;
	return total.x * total.y;
}

//Explicit Instantiation
#define NN_TEXTURE_BUFFER(TYPEA, TYPEB) \
	template class STP_API SuperTerrainPlus::STPNearestNeighbourTextureBuffer<TYPEA, STPNearestNeighbourTextureBufferMemoryMode::TYPEB>
NN_TEXTURE_BUFFER(float, WriteOnly);
NN_TEXTURE_BUFFER(float, ReadWrite);
NN_TEXTURE_BUFFER(Sample, ReadOnly);
NN_TEXTURE_BUFFER(unsigned short, WriteOnly);