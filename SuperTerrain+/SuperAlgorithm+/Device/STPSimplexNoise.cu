#include <SuperAlgorithm+/Device/STPSimplexNoise.cuh>

#include <cassert>

using namespace SuperTerrainPlus::STPAlgorithm;

constexpr static float F2 = 0.3660254038;// 0.5 * (sqrt(3.0) - 1.0)
constexpr static float G2 = 0.2113248654;// (3.0 - sqrt(3.0)) / 6.0
constexpr static float H2 = -1.0 + 2.0 * G2;

__device__ __forceinline__ constexpr static float dot2D(float, float, float, float);

__device__ __forceinline__ constexpr static int floori(float);

__device__ STPSimplexNoise::STPSimplexNoise(const STPPermutation& permutation) : Permutation(permutation) {

}

__device__ STPSimplexNoise::~STPSimplexNoise() {

}

__device__ __inline__ int STPSimplexNoise::perm(int index) const {
	//device memory can be accessed in device directly
	return static_cast<int>(this->Permutation.Permutation[index]);
}

__device__ __inline__ float STPSimplexNoise::grad2D(int index, int component) const {
	return this->Permutation.Gradient2D[index * 2 + component];
}

__device__ float STPSimplexNoise::simplex2D(float x, float y) const {
	//noise distributions from the 3 corners
	//and the distance to three corners
	float corner[3], dst_x[3], dst_y[3], weight[3];

	//coordinate system skewing to determine which simplex cell we are in
	const float s = (x + y) * F2;
	const int i = floori(x + s), //add then floor (round down)
		j = floori(y + s); //(i,j) space
	
	//unskewing the cells
	const float t = (i + j) * G2,
		X0 = i - t, //unskweing the cell origin back to (x,y) space
		Y0 = j - t;
	dst_x[0] = x - X0; //The distance from the cell origin in (x,y) space
	dst_y[0] = y - Y0;

	//In 2D, simplex is just a triangle (equilateral)
	//Determine which simplex we are in
	//offsets for the middle corner of simplex in (i,j) space
	//lower triangle, XY order: (0,0)->(1,0)->(1,1)
	//otherwise upper triangle, YX order: (0,0)->(0,1)->(1,1)
	const int i1 = (dst_x[0] > dst_y[0]) ? 1 : 0,
		j1 = 1 - i1;

	// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	// c = (3-sqrt(3))/6
	
	dst_x[1] = dst_x[0] - i1 + G2; //Now offset the middle corner in the unskewed space (x,y)
	dst_y[1] = dst_y[0] - j1 + G2;
	dst_x[2] = dst_x[0] + H2; //Offset for the last corner in (x,y)
	dst_y[2] = dst_y[0] + H2;

	const int grad2DSize = static_cast<int>(this->Permutation.Gradient2DSize);
	//Read the hashed gradient indices of the three simplex corner
	const int ii = i & 255,
		jj = j & 255;
	int grad_i[3];
	grad_i[0] = this->perm(ii + this->perm(jj)) % grad2DSize;
	grad_i[1] = this->perm(ii + i1 + this->perm(jj + j1)) % grad2DSize;
	grad_i[2] = this->perm(ii + 1 + this->perm(jj + 1)) % grad2DSize;

	//Calcultate the weight from 3 corners
	for (int vertex = 0; vertex < 3; vertex++) {
		weight[vertex] = 0.5f - dst_x[vertex] * dst_x[vertex] - dst_y[vertex] * dst_y[vertex];
		if (weight[vertex] <= 0.0f) {
			corner[vertex] = 0.0f;
		}
		else {
			weight[vertex] *= weight[vertex];
			corner[vertex] = weight[vertex] * weight[vertex]
				* dot2D(this->grad2D(grad_i[vertex], 0), this->grad2D(grad_i[vertex], 1), dst_x[vertex], dst_y[vertex]);
		}
	}

	//add all the weights together
	//scale the result to [-1,1]
	return 70.0f * (corner[0] + corner[1] + corner[2]);
}

__device__ float3 STPSimplexNoise::simplex2DFractal(float x, float y, STPFractalSimplexInformation& desc) const {
	//The min and max (y and z component) indicates the range of the multi-phased simplex function, not the range of the output texture
	float3 fractal = make_float3(0.0f, 0.0f, 0.0f);

	//multiple phases of noise
	for (unsigned int i = 0u; i < desc.Octave; i++) {
		const float sampleX = ((x - desc.HalfDimension.x) + desc.Offset.x) / desc.Scale * desc.Frequency, //subtract the half width and height can make the scaling focus at the center
			sampleY = ((y - desc.HalfDimension.y) + desc.Offset.y) / desc.Scale * desc.Frequency;//since the y is inverted we want to filp it over
		fractal.x += this->simplex2D(sampleX, sampleY) * desc.Amplitude;

		//calculate the min and max
		fractal.y -= desc.Amplitude;
		fractal.z += desc.Amplitude;
		//scale the parameters
		desc.Amplitude *= desc.Persistence;
		desc.Frequency *= desc.Lacunarity;
	}

	return fractal;
}

__device__ __forceinline__ constexpr static float dot2D(float v1x, float v1y, float v2x, float v2y) {
	return v1x * v2x + v1y * v2y;
}

__device__ __forceinline__ constexpr static int floori(float x) {
	return x > 0.0f ? static_cast<int>(x) : static_cast<int>(x - 1.0f);
}