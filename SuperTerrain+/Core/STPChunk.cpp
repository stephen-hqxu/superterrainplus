#include <World/Chunk/STPChunk.h>

using glm::vec2;
using glm::uvec2;
using glm::ivec2;
using glm::vec3;

using std::unique_ptr;
using std::make_unique;
using std::atomic_init;
using std::atomic_load;
using std::atomic_store;
using std::ostream;
using std::istream;
using std::ios_base;

using namespace SuperTerrainPlus;

STPChunk::STPChunk(uvec2 size, bool initialise) : PixelSize(size) {
	if (initialise) {
		const int num_pixel = size.x * size.y;
		//heightmap is R32F format
		this->Heightmap = make_unique<float[]>(num_pixel);
		//biomemap is R16UI format
		this->Biomemap = make_unique<STPDiversity::Sample[]>(num_pixel);
		//rendering buffer is RGBA16 format, contains heightmap and normalmap
		this->TerrainRenderingBuffer = make_unique<unsigned short[]>(num_pixel * 4);
	}

	atomic_init<STPChunkState>(&this->State, STPChunkState::Empty);
	atomic_init<bool>(&this->inUsed, false);
}

STPChunk::~STPChunk() {
	while (atomic_load<bool>(&this->inUsed)) {
		//make sure the chunk is not in used, and all previous tasks are finished
	}

	//array deleted by smart ptr
}

bool STPChunk::isOccupied() const {
	return atomic_load(&this->inUsed);
}

void STPChunk::markOccupancy(bool val) {
	atomic_store(&this->inUsed, val);
}

STPChunk::STPChunkState STPChunk::getChunkState() const {
	return atomic_load(&this->State);
}

void STPChunk::markChunkState(STPChunkState state) {
	atomic_store(&this->State, state);
}

float* STPChunk::getHeightmap() {
	return this->Heightmap.get();
}

unsigned short* STPChunk::getRenderingBuffer() {
	return this->TerrainRenderingBuffer.get();
}

STPDiversity::Sample* STPChunk::getBiomemap() {
	return this->Biomemap.get();
}

const uvec2& STPChunk::getSize() const {
	return this->PixelSize;
}

vec2 STPChunk::getChunkPosition(vec3 cameraPos, uvec2 chunkSize, float scaling) noexcept {
	const vec2 chunkSize_int = vec2(scaling * static_cast<int>(chunkSize.x), scaling * static_cast<int>(chunkSize.y));
	return vec2(glm::floor(cameraPos.x / chunkSize_int.x) * chunkSize_int.x, glm::floor(cameraPos.z / chunkSize_int.y) * chunkSize_int.y);
}

vec2 STPChunk::offsetChunk(vec2 chunkPos, uvec2 chunkSize, ivec2 offset, float scaling) noexcept {
	return chunkPos + vec2(scaling * static_cast<int>(chunkSize.x) * offset.x, scaling * static_cast<int>(chunkSize.y) * offset.y);
}

STPChunk::STPChunkPositionCache STPChunk::getRegion(vec2 centerPos, uvec2 chunkSize, uvec2 regionSize, float scaling) noexcept {
	const unsigned int num_chunk = regionSize.x * regionSize.y;
	//We need to calculate the starting unit plane, i.e., the unit plane of the top-left corner of the entire rendered terrain
	//We don't need y, all the plane will be aligned at the same height, so it contains x and z position
	//Base position is negative since we want the camera locates above the center of the entire rendered plane
	const vec2 base_position = STPChunk::offsetChunk(centerPos, chunkSize, ivec2(-glm::floor(regionSize.x / 2.0f), -glm::floor(regionSize.y / 2.0f)), scaling);

	STPChunkPositionCache results;
	results.reserve(num_chunk);
	//get the result
	for (unsigned int i = 0u; i < num_chunk; i++) {
		//Calculate the position for each chunk
		//Note that the chunk_position is not the basechunkpos, the former one is refered to the relative coordinate to the entire terrain, i.e., local chunk coordinate.
		//The latter one refers to the world coordinate

		//Basically it converts 1D chunk ID to 2D local chunk position, btw chunk position must be a positive integer
		const uvec2 local_chunk_offset = uvec2(i % regionSize.x, static_cast<unsigned int>(glm::floor<unsigned int>(i / regionSize.y)));
		//Then convert local to world coordinate
		const vec2 basePos = STPChunk::offsetChunk(base_position, chunkSize, local_chunk_offset, scaling);

		//arranged from top-left to bottom right
		results.push_back(basePos);
	}

	return results;
}

namespace SuperTerrainPlus {

	//TODO: serialisation and deserialisation don't store normalmap hence rendering buffer
	ostream& operator<<(ostream& output, const STPChunk* const chunk) {
		output.seekp(ios_base::beg);
		//write identifier
		output.write(reinterpret_cast<const char*>(&STPChunk::IDENTIFIER), sizeof(unsigned long long));
		//write serial version number
		output.write(reinterpret_cast<const char*>(&STPChunk::SERIAL_VERSION), sizeof(unsigned short));

		//main content
		//write chunk state
		const char state = static_cast<const char>(atomic_load(&chunk->State));
		output.write(&state, sizeof(char));
		//write pixel size, x and y
		output.write(reinterpret_cast<const char*>(&chunk->PixelSize.x), sizeof(unsigned int));
		output.write(reinterpret_cast<const char*>(&chunk->PixelSize.y), sizeof(unsigned int));
		const unsigned int size = chunk->PixelSize.x * chunk->PixelSize.y;
		//write heightmap, 1 channel
		output.write(reinterpret_cast<const char*>(chunk->Heightmap.get()), sizeof(float) * size);

		//finish up
		output.flush();
		return output;
	}

	istream& operator>>(istream& input, STPChunk*& chunk) {
		input.seekg(ios_base::beg);
		//read identifier
		unsigned long long id;
		input.read(reinterpret_cast<char*>(&id), sizeof(unsigned long long));
		//id check
		if ((~id & STPChunk::IDENTIFIER) != 0ull) {
			throw STPSerialisationException("File format is not defined as a valid STPChunk object");
		}
		//read serial version number
		unsigned short version;
		input.read(reinterpret_cast<char*>(&version), sizeof(unsigned short));
		//version check
		if (((~version & STPChunk::SERIAL_VERSION) >> 8u) != 0u) {
			//if the major version doesn't match, stop parsing
			throw STPSerialisationException("Serialisation protocol version is deprecated");
		}

		//main content
		//read chunk state
		char state;
		input.read(&state, sizeof(char));
		//read pixel size, x and y
		uvec2 pix_size;
		input.read(reinterpret_cast<char*>(&pix_size.x), sizeof(unsigned int));
		input.read(reinterpret_cast<char*>(&pix_size.y), sizeof(unsigned int));
		const unsigned int size = pix_size.x * pix_size.y;
		//allocation
		chunk = new STPChunk(pix_size, true);
		//read heightmap, 1 channel
		input.read(reinterpret_cast<char*>(chunk->Heightmap.get()), sizeof(float) * size);
		//chunk state
		atomic_store(&chunk->State, static_cast<STPChunk::STPChunkState>(state));

		//finish up
		input.sync();
		return input;
	}
}