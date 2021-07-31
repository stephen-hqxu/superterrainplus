#pragma once
#include <STPSimplexNoise.cuh>

#include <cassert>
//Permutation
#include "../STPPermutation.hpp"

using namespace SuperTerrainPlus::STPCompute;

constexpr static double csqrt(double x) {
	assert(x >= 0);
	if (x == 0) {
		return 0;
	}

	//For small numbers small iteration is enough for a good approximation
	constexpr unsigned char Iteration = 5u;

	//Newton-Rhapson's method
	double guess = x;
	for (unsigned char i = 0u; i < Iteration; i++) {
		guess -= (guess * guess - x) / (2 * guess);
	}
	return guess;
}
constexpr static __constant__ double F2 = 0.5 * (csqrt(3.0) - 1.0);
constexpr static __constant__ double G2 = (3.0 - csqrt(3.0)) / 6.0;
constexpr static __constant__ double H2 = -1.0 + 2.0 * G2; // -1.0 + 2.0 * G2;

/**
 * @brief Perform 2D vector dot product
 * @param v1x vector 1 x
 * @param v1y vector 1 y
 * @param v2x vector 2 x
 * @param v2y vector 2 y
 * @return The result
*/
__device__ __forceinline__ double dot2D(double, double, double, double);

__device__ STPSimplexNoise::STPSimplexNoise(const STPPermutation_t permutation) : Permutation(permutation) {

}

__device__ STPSimplexNoise::~STPSimplexNoise() {

}

__device__ int STPSimplexNoise::perm(int index) const {
	//device memory can be accessed in device directly
	return static_cast<int>(this->Permutation->Permutation[index]);
}

__device__ double STPSimplexNoise::grad2D(int index, int component) const {
	//convert two int bits to one double bit
	return this->Permutation->Gradient2D[index * 2 + component];
}

__device__ unsigned int STPSimplexNoise::grad2D_size() const {
	return this->Permutation->Gradient2DSize;
}

__device__ float STPSimplexNoise::simplex2D(float x, float y) const {
	//noise distributions from the 3 corners
	//and the distance to three corners
	float corner[3], dst_x[3], dst_y[3], weight[3];

	//coordinate system skewing to determine which simplex cell we are in
	const float hairy_factor2D = dot2D(x, y, F2, F2);
	const int i = static_cast<int>(floorf(x + hairy_factor2D)), //add then floor (round down)
		j = static_cast<int>(floorf(y + hairy_factor2D)); //(i,j) space
	
	//unskewing the cells
	const float original_factor2D = dot2D(i, j, G2, G2),
		X0 = i - original_factor2D, //unskweing the cell origin back to (x,y) space
		Y0 = j - original_factor2D;
	dst_x[0] = x - X0; //The distance from the cell origin in (x,y) space
	dst_y[0] = y - Y0;

	//In 2D, simplex is just a triangle (equilateral)
	//Determine which simplex we are in
	//offsets for the middle corner of simplex in (i,j) space
	//lower triangle, XY order: (0,0)->(1,0)->(1,1)
	//otherwise upper triangle, YX order: (0,0)->(0,1)->(1,1)
	const int offseti1 = (dst_x[0] > dst_y[0]) ? 1 : 0,
	offsetj1 = 1 - offseti1;

	// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	// c = (3-sqrt(3))/6
	
	dst_x[1] = dst_x[0] - offseti1 + G2; //Now offset the middle corner in the unskewed space (x,y)
	dst_y[1] = dst_y[0] - offsetj1 + G2;
	dst_x[2] = dst_x[0] + H2; //Offset for the last corner in (x,y)
	dst_y[2] = dst_y[0] + H2;

	//Read the hashed gradient indices of the three simplex corner
	const int ii = i & 255,
		jj = j & 255;
	int grad_i[3];
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

__device__ __forceinline__ double dot2D(double v1x, double v1y, double v2x, double v2y) {
	return v1x * v2x + v1y * v2y;
}