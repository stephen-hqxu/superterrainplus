#pragma once
#pragma warning(disable:26495)//Alarm for un-init variables
#include "STPRainDrop.cuh"

using namespace SuperTerrainPlus::STPCompute;

__device__ STPRainDrop::STPRainDrop(float2 position, float WaterVolume, float MovementSpeed) {
	//copy the contructor
	this->raindrop_pos = position;
	this->raindrop_dir = make_float2(0.0f, 0.0f);
	//They are the initial values.
	this->volume = WaterVolume;
	this->speed = MovementSpeed;
}

__device__ STPRainDrop::~STPRainDrop() {

}

__host__ STPRainDrop::STPFreeSlipManager::STPFreeSlipManager(float* heightmap, unsigned int* index, uint2 range, uint2 mapSize)
	: Dimension(mapSize), FreeSlipChunk(range), FreeSlipRange(make_uint2(range.x * mapSize.x, range.y * mapSize.y)) {
	this->Heightmap = heightmap;
	this->Index = index;
}

__host__ STPRainDrop::STPFreeSlipManager::~STPFreeSlipManager() {

}

__device__ float& STPRainDrop::STPFreeSlipManager::operator[](unsigned int global) {
	return this->Heightmap[this->Index[global]];
}

__device__ float3 STPRainDrop::calcHeightGradients(STPFreeSlipManager& map) {
	//result
	float3 height_gradients;

	const uint2 rounded_pos = make_uint2(static_cast<unsigned int>(this->raindrop_pos.x), static_cast<unsigned int>(this->raindrop_pos.y));
	//calculate drop's offset inside the cell (0,0) and (1,1)
	const float2 cell_corner = make_float2(this->raindrop_pos.x - rounded_pos.x, this->raindrop_pos.y - rounded_pos.y);
	//calculate the heights of the 4 nodes of the droplet's cell
	const unsigned int nodebaseIndex = rounded_pos.y * map.FreeSlipRange.x + rounded_pos.x;//The position on the map of the local (0,0) cell
	const float4 heights = make_float4(
		map[nodebaseIndex], // (0,0)
		map[nodebaseIndex + 1], // (1,0)
		map[nodebaseIndex + map.FreeSlipRange.x], // (1,0)
		map[nodebaseIndex + map.FreeSlipRange.x + 1] // (1,1)
	);

	//calculate height with bilinear interpolation of the heights of the nodes of the cell
	height_gradients.x =
		  heights.x * (1 - cell_corner.x) * (1 - cell_corner.y)
		+ heights.y * cell_corner.x * (1 - cell_corner.y)
		+ heights.z * (1 - cell_corner.x) * cell_corner.y
		+ heights.w * cell_corner.x * cell_corner.y;
	//calculate droplet's direction of flow with bilinear interpolation of height difference along the edge
	height_gradients.y = (heights.y - heights.x) * (1 - cell_corner.y) + (heights.w - heights.z) * cell_corner.y;
	height_gradients.z = (heights.z - heights.x) * (1 - cell_corner.x) + (heights.w - heights.y) * cell_corner.x;

	return height_gradients;
}

__device__ float STPRainDrop::getCurrentVolume() const {
	return this->volume;
}

__device__ void STPRainDrop::Erode(const STPSettings::STPRainDropSettings* const settings, STPFreeSlipManager& map) {
	//Err, this algorithm is gonna be sick... But let's start!
	//Rain drop is still alive, continue descending...
	while (this->volume >= settings->minWaterVolume) {
		//The position of droplet on the map index
		unsigned int mapIndex = static_cast<unsigned int>(this->raindrop_pos.y) * map.FreeSlipRange.x + static_cast<unsigned int>(this->raindrop_pos.x);
		//calculate the offset of the droplet inside cell (0,0) and cell (1,1)
		float2 offset_cell = make_float2(this->raindrop_pos.x - static_cast<int>(this->raindrop_pos.x), this->raindrop_pos.y - static_cast<int>(this->raindrop_pos.y));
		//check if the particle is not accelerating and is it surrounded by a lot of other particles

		//calculate droplet's height and the direction of flow with bilinear interpolation of surrounding heights
		const float3 height_gradients = STPRainDrop::calcHeightGradients(map);
		
		//update droplet's position and direction
		this->raindrop_dir.x = this->raindrop_dir.x * settings->Inertia - height_gradients.y * (1.0f - settings->Inertia);
		this->raindrop_dir.y = this->raindrop_dir.y * settings->Inertia - height_gradients.z * (1.0f - settings->Inertia);
		//normalise the direction and update the position and direction, (move position 1 unit regardless of speed)
		float length = fmaxf(0.01f, sqrtf(this->raindrop_dir.x * this->raindrop_dir.x + this->raindrop_dir.y * this->raindrop_dir.y));
		this->raindrop_dir.x /= length;
		this->raindrop_dir.y /= length;
		this->raindrop_pos.x += this->raindrop_dir.x;
		this->raindrop_pos.y += this->raindrop_dir.y;

		//check if the raindrop brushing range falls out of the map
		if ((this->raindrop_dir.x == 0.0f && this->raindrop_dir.y == 0.0f) 
			|| this->raindrop_pos.x < (settings->getErosionBrushRadius() * 1.0f) || this->raindrop_pos.x >= 1.0f * map.FreeSlipRange.x - settings->getErosionBrushRadius()
			|| this->raindrop_pos.y < (settings->getErosionBrushRadius() * 1.0f) || this->raindrop_pos.y >= 1.0f * map.FreeSlipRange.y - settings->getErosionBrushRadius()) {
			//ending the life of this poor raindrop
			this->volume = 0.0f;
			this->sediment = 0.0f;
			break;
		}

		//find the new height and calculate the delta height
		const float deltaHeight = STPRainDrop::calcHeightGradients(map).x - height_gradients.x;

		//calculate droplet's sediment capacity (higher when moving fast down a slop and contains a lot of water)
		const float sedimentCapacity = fmaxf(-deltaHeight * this->speed * this->volume * settings->SedimentCapacityFactor, settings->minSedimentCapacity);
		
		//if carrying more sediment than capacity, or it's flowing uphill
		if (this->sediment > sedimentCapacity || deltaHeight > 0.0f) {
			//If flowing uphill (deltaheight > 0) try to fill up the current height, otherwise deposit a fraction of the excess sediment
			float depositAmount = (deltaHeight > 0.0f) ? fminf(deltaHeight, this->sediment) : (this->sediment - sedimentCapacity) * settings->DepositSpeed ;
			this->sediment -= depositAmount;

			//add the sediment to the four nodes of the current cell using bilinear interpolation
			//deposition is not distributed over a radius (like erosion) so that it can fill small pits :)
			map[mapIndex] += depositAmount * (1.0f - offset_cell.x) * (1.0f - offset_cell.y);
			map[mapIndex + 1] += depositAmount * offset_cell.x * (1.0f - offset_cell.y);
			map[mapIndex + map.FreeSlipRange.x] += depositAmount * (1.0f - offset_cell.x) * offset_cell.y;
			map[mapIndex + map.FreeSlipRange.x + 1] += depositAmount * offset_cell.x *  offset_cell.y;
		}
		else {
			//erode a fraction of the droplet's current carry capacity
			//clamp the erosion to the change in height so that it doesn't dig a hole in the terrain behind the droplet
			const float erodeAmout = fminf((sedimentCapacity - this->sediment) * settings->ErodeSpeed, -deltaHeight);

			//use erode brush to erode from all nodes inside the droplet's erode radius
			for (unsigned int brushPointIndex = 0u; brushPointIndex < settings->getErosionBrushSize(); brushPointIndex++) {
				const int brush_index = __ldg(settings->ErosionBrushIndices + brushPointIndex);
				const float brush_weight = __ldg(settings->ErosionBrushWeights + brushPointIndex);

				int erodeIndex = mapIndex + brush_index;
				float weightederodeAmout = erodeAmout * brush_weight;
				float deltaSediment = (map[erodeIndex] < weightederodeAmout) ? map[erodeIndex] : weightederodeAmout;
				//erode the map
				map[erodeIndex] -= deltaSediment;
				this->sediment += deltaSediment;
			}
		}
		//update droplet's speed and water content
		this->speed = sqrtf(fmaxf(0.0f, this->speed * this->speed + deltaHeight * settings->Gravity));//Newton's 2nd Law
		this->speed *= 1.0f - settings->Friction;//Newton's Friction Equation
		this->volume *= (1.0f - settings->EvaporateSpeed);

	}
}
#pragma warning(default:26495)