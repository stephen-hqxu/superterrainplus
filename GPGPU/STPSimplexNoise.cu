#pragma once
#include "STPSimplexNoise.cuh"
#include <stdexcept>

using namespace SuperTerrainPlus::STPCompute;

bool init = false;
__constant__ double F2[1]; // 0.5 * (static_cast<double>(sqrt(3.0)) - 1.0);
__constant__ double G2[1]; // (3.0 - static_cast<double>(sqrt(3.0))) / 6.0;
__constant__ double H2[1]; // -1.0 + 2.0 * G2;

/**
 * @brief Perform 2D vector dot product
 * @param v1x vector 1 x
 * @param v1y vector 1 y
 * @param v2x vector 2 x
 * @param v2y vector 2 y
 * @return The result
*/
__device__ __forceinline__ float dot2D(float, float, float, float);

__host__ STPSimplexNoise::STPSimplexNoise(const STPSettings::STPSimplexNoiseSettings* const noise_settings)
	: STPPermutationsGenerator(noise_settings->Seed, noise_settings->Distribution, noise_settings->Offset) {
	if (!init) {
		throw std::runtime_error("Simplex noise generator has been initialised");
	}
}

__host__ STPSimplexNoise::~STPSimplexNoise() {

}

__host__ bool STPSimplexNoise::initialise() {
	bool status = true;
	const double f2 = 0.366025403784439;
	const double g2 = 0.211324865405187;
	const double h2 = -0.577350269189626;

	status &= cudaSuccess == cudaMemcpyToSymbol(F2, &f2, sizeof(double), 0ull, cudaMemcpyHostToDevice);
	status &= cudaSuccess == cudaMemcpyToSymbol(G2, &g2, sizeof(double), 0ull, cudaMemcpyHostToDevice);
	status &= cudaSuccess == cudaMemcpyToSymbol(H2, &h2, sizeof(double), 0ull, cudaMemcpyHostToDevice);

	init = status;
	return status;
}

__device__ float STPSimplexNoise::simplex2D(float x, float y) const {
	//noise distributions from the 3 corners
	//and the distance to three corners
	float corner[3], dst_x[3], dst_y[3], weight[3];

	//coordinate system skewing to determine which simplex cell we are in
	float hairy_factor2D = dot2D(x, y, *F2, *F2);
	int i = static_cast<int>(floor(x + hairy_factor2D)), //add then floor (round down)
		j = static_cast<int>(floor(y + hairy_factor2D)); //(i,j) space
	
	//unskewing the cells
	float original_factor2D = dot2D(i, j, *G2, *G2),
		X0 = i - original_factor2D, //unskweing the cell origin back to (x,y) space
		Y0 = j - original_factor2D;
	dst_x[0] = x - X0; //The distance from the cell origin in (x,y) space
	dst_y[0] = y - Y0;

	//In 2D, simplex is just a triangle (equilateral)
	//Determine which simplex we are in
	int offseti1, offsetj1;//offsets for the middle corner of simplex in (i,j) space
	//lower triangle, XY order: (0,0)->(1,0)->(1,1)
	//otherwise upper triangle, YX order: (0,0)->(0,1)->(1,1)
	offseti1 = (dst_x[0] > dst_y[0]) ? 1 : 0;
	offsetj1 = 1 - offseti1;

	// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	// c = (3-sqrt(3))/6
	
	dst_x[1] = dst_x[0] - offseti1 + *G2; //Now offset the middle corner in the unskewed space (x,y)
	dst_y[1] = dst_y[0] - offsetj1 + *G2;
	dst_x[2] = dst_x[0] + *H2; //Offset for the last corner in (x,y)
	dst_y[2] = dst_y[0] + *H2;

	//Read the hashed gradient indices of the three simplex corner
	int grad_i[3];
	int ii = i & 255,
		jj = j & 255;
	grad_i[0] = static_cast<int>(fmodf(this->perm(ii + this->perm(jj)), this->grad2D_size()));
	grad_i[1] = static_cast<int>(fmodf(this->perm(ii + offseti1 + this->perm(jj + offsetj1)), this->grad2D_size()));
	grad_i[2] = static_cast<int>(fmodf(this->perm(ii + 1 + this->perm(jj + 1)), this->grad2D_size()));

	//Calcultate the weight from 3 corners
	for (int vertex = 0; vertex < 3; vertex++) {
		weight[vertex] = 0.5f - dst_x[vertex] * dst_x[vertex] - dst_y[vertex] * dst_y[vertex];
		if (weight[vertex] <= 0.0f) {
			corner[vertex] = 0.0f;
		}
		else {
			weight[vertex] *= weight[vertex];
			corner[vertex] = weight[vertex] * weight[vertex] * dot2D(this->grad2D(grad_i[vertex], 0), this->grad2D(grad_i[vertex], 1), dst_x[vertex], dst_y[vertex]);
		}
	}

	//add all the weights together
	//scale the result to [-1,1]
	return 70.0f * (corner[0] + corner[1] + corner[2]);
}

__device__ __forceinline__ float dot2D(float v1x, float v1y, float v2x, float v2y) {
	return v1x * v2x + v1y * v2y;
}