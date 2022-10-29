#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

//CUDA Device Parameters
#include <device_launch_parameters.h>

using namespace SuperTerrainPlus;

//GLM
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;

__device__ STPRainDrop::STPRainDrop(vec2 position, float WaterVolume, float MovementSpeed, uvec2 dimension) : 
	Position(position), Direction(0.0f), Speed(MovementSpeed), Volume(WaterVolume), Dimension(dimension) {

}

__device__ STPRainDrop::~STPRainDrop() {

}

__device__ vec3 STPRainDrop::calcHeightGradients(const float* map) const {
	const unsigned int rowCount = this->Dimension.x;
	//result
	vec3 height_gradients;

	const uvec2 rounded_pos = static_cast<uvec2>(this->Position);
	//calculate drop's offset inside the cell (0,0) and (1,1)
	const vec2 cell_corner = this->Position - static_cast<vec2>(rounded_pos);
	//calculate the heights of the 4 nodes of the droplet's cell
	const unsigned int nodebaseIndex = rounded_pos.y * rowCount + rounded_pos.x;//The position on the map of the local (0,0) cell
	const vec4 heights(
		map[nodebaseIndex], // (0,0)
		map[nodebaseIndex + 1], // (1,0)
		map[nodebaseIndex + rowCount], // (1,0)
		map[nodebaseIndex + rowCount + 1] // (1,1)
	);

	//calculate height with bilinear interpolation of the heights of the nodes of the cell
	height_gradients.x =
		  heights.x * (1 - cell_corner.x) * (1 - cell_corner.y)
		+ heights.y * cell_corner.x * (1 - cell_corner.y)
		+ heights.z * (1 - cell_corner.x) * cell_corner.y
		+ heights.w * cell_corner.x * cell_corner.y;
	//calculate droplet's direction of flow with bilinear interpolation of height difference along the edge
	height_gradients.y = glm::mix(heights.y - heights.x, heights.w - heights.z, cell_corner.y);
	height_gradients.z = glm::mix(heights.z - heights.x, heights.w - heights.y, cell_corner.x);

	return height_gradients;
}

__device__ void STPRainDrop::operator()(float* map, const STPEnvironment::STPRainDropSetting& settings, const STPErosionBrush& brush) {
	const auto [raw_brushIndex, raw_brushWeight, brushSize] = brush;
	const unsigned int brushRadius = settings.ErosionBrushRadius;

	//Cache erosion brush to shared memory
	//Erosion brush indices then weights
	extern __shared__ unsigned char ErosionBrush[];
	int* brushIndices = reinterpret_cast<int*>(ErosionBrush);
	float* brushWeights = reinterpret_cast<float*>(ErosionBrush + sizeof(int) * brushSize);
	{
		unsigned int iteration = 0u;
		while (iteration < brushSize) {
			unsigned int idx = threadIdx.x + iteration;
			if (idx < brushSize) {
				//check and make sure index is not out of bound
				//otherwise we can utilise most threads and copy everything in parallel
				brushIndices[idx] = raw_brushIndex[idx];
				brushWeights[idx] = raw_brushWeight[idx];
			}
			//if erosion brush size is greater than number of thread in a block
			//we need to warp around and reuse some threads to finish the rests
			iteration += blockDim.x;
		}
	}
	__syncthreads();

	//Rain drop is still alive, continue descending...
	while (this->Volume >= settings.minWaterVolume) {
		//The position of droplet on the map index
		const unsigned int mapIndex = static_cast<unsigned int>(this->Position.y) * this->Dimension.x
			+ static_cast<unsigned int>(this->Position.x);
		//calculate the offset of the droplet inside cell (0,0) and cell (1,1)
		const vec2 offset_cell = this->Position - static_cast<vec2>(static_cast<ivec2>(this->Position));
		//check if the particle is not accelerating and is it surrounded by a lot of other particles

		//calculate droplet's height and the direction of flow with bilinear interpolation of surrounding heights
		const vec3 height_gradients = STPRainDrop::calcHeightGradients(map);
		
		//update droplet's position and direction
		this->Direction = glm::mix(-vec2(height_gradients.y, height_gradients.z), this->Direction, settings.Inertia);
		//normalise the direction and update the position and direction, (move position 1 unit regardless of speed)
		//clamp the length to handle division by zero instead of using the glm::normalize directly
		const float length = glm::length(this->Direction);
		if (length != 0.0f) {
			this->Direction /= length;
		}
		this->Position += this->Direction;

		//check if the raindrop brushing range falls out of the map
		if ((this->Direction.x == 0.0f && this->Direction.y == 0.0f)
			|| this->Position.x < (brushRadius * 1.0f)
			|| this->Position.x >= 1.0f * this->Dimension.x - brushRadius
			|| this->Position.y < (brushRadius * 1.0f)
			|| this->Position.y >= 1.0f * this->Dimension.y - brushRadius) {
			//ending the life of this poor raindrop
			this->Volume = 0.0f;
			this->Sediment = 0.0f;
			break;
		}

		//find the new height and calculate the delta height
		const float deltaHeight = STPRainDrop::calcHeightGradients(map).x - height_gradients.x;

		//calculate droplet's sediment capacity (higher when moving fast down a slop and contains a lot of water)
		const float sedimentCapacity = fmaxf(-deltaHeight * this->Speed * this->Volume * settings.SedimentCapacityFactor, settings.minSedimentCapacity);
		
		//if carrying more sediment than capacity, or it's flowing uphill
		if (this->Sediment > sedimentCapacity || deltaHeight > 0.0f) {
			//If flowing uphill (delta height > 0) try to fill up the current height, otherwise deposit a fraction of the excess sediment
			const float depositAmount = (deltaHeight > 0.0f)
				? fminf(deltaHeight, this->Sediment)
				: (this->Sediment - sedimentCapacity) * settings.DepositSpeed;
			this->Sediment -= depositAmount;

			//add the sediment to the four nodes of the current cell using bilinear interpolation
			//deposition is not distributed over a radius (like erosion) so that it can fill small pits :)
#ifndef NDEBUG
			//The current erosion algorithm is imperfect, race condition occurs when multiple raindrop collides into each other at the same time.
			//Although the chance might be small when the heightmap is large, but we still need to prevent it somehow.
			//This can be prevented by using atomic operation, however it is exceptionally slow in debug build.
			//So we essentially trade data safety with performance on debug build.
			map[mapIndex] += depositAmount * (1.0f - offset_cell.x) * (1.0f - offset_cell.y);
			map[mapIndex + 1] += depositAmount * offset_cell.x * (1.0f - offset_cell.y);
			map[mapIndex + this->Dimension.x] += depositAmount * (1.0f - offset_cell.x) * offset_cell.y;
			map[mapIndex + this->Dimension.x + 1] += depositAmount * offset_cell.x *  offset_cell.y;
#else
			//On release build, it is marginally faster to use atomic instruction, so we enable it.
			atomicAdd(map + mapIndex, depositAmount * (1.0f - offset_cell.x) * (1.0f - offset_cell.y));
			atomicAdd(map + mapIndex + 1, depositAmount * offset_cell.x * (1.0f - offset_cell.y));
			atomicAdd(map + mapIndex + this->Dimension.x, depositAmount * (1.0f - offset_cell.x) * offset_cell.y);
			atomicAdd(map + mapIndex + this->Dimension.x + 1, depositAmount * offset_cell.x * offset_cell.y);
#endif
		}
		else {
			//erode a fraction of the droplet's current carry capacity
			//clamp the erosion to the change in height so that it doesn't dig a hole in the terrain behind the droplet
			const float erodeAmout = fminf((sedimentCapacity - this->Sediment) * settings.ErodeSpeed, -deltaHeight);

			//use erode brush to erode from all nodes inside the droplet's erode radius
			for (unsigned int brushPointIndex = 0u; brushPointIndex < brushSize; brushPointIndex++) {
				const unsigned int erodeIndex = mapIndex + brushIndices[brushPointIndex];
				const float weightederodeAmout = erodeAmout * brushWeights[brushPointIndex];
				const float deltaSediment = (map[erodeIndex] < weightederodeAmout) ? map[erodeIndex] : weightederodeAmout;
				//erode the map
#ifndef NDEBUG
				map[erodeIndex] -= deltaSediment;
#else
				atomicAdd(map + erodeIndex, -deltaSediment);
#endif
				this->Sediment += deltaSediment;
			}
		}
		//update droplet's speed and water content
		this->Speed = sqrtf(fmaxf(0.0f, this->Speed * this->Speed + deltaHeight * settings.Gravity));//Newton's 2nd Law
		this->Speed *= 1.0f - settings.Friction;//Newton's Friction Equation
		this->Volume *= (1.0f - settings.EvaporateSpeed);

	}
}