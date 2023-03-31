#include <SuperAlgorithm+Device/STPSimplexNoise.cuh>

using namespace SuperTerrainPlus::STPAlgorithm;

constexpr static float F2 = 0.3660254038f;// 0.5 * (sqrt(3.0) - 1.0)
constexpr static float G2 = 0.2113248654f;// (3.0 - sqrt(3.0)) / 6.0
constexpr static float H2 = -1.0f + 2.0f * G2;

__device__ __forceinline__ static float dot2D(const float v1x, const float v1y, const float v2x, const float v2y) {
	return v1x * v2x + v1y * v2y;
}

//a fast floor? I didn't test how fast it is compared to built-in maths function
__device__ __forceinline__ static int floori(const float x) {
	return x > 0.0f ? static_cast<int>(x) : static_cast<int>(x - 1.0f);
}

__device__ float STPSimplexNoise::simplex2D(const STPPermutation& permutation, const float x, const float y) {
	const auto grad2D = [perm_grad2D = permutation.Gradient2D] __device__(const unsigned int index, const unsigned int component) -> float {
		return perm_grad2D[index * 2u + component];
	};

	//noise distributions from the 3 corners
	//and the distance to three corners
	float corner[3], dst_x[3], dst_y[3], weight[3];

	//coordinate system skewing to determine which simplex cell we are in
	const float s = (x + y) * F2;
	const int i = floori(x + s), //add then floor (round down)
		j = floori(y + s); //(i,j) space
	
	//un-skewing the cells
	const float t = (i + j) * G2,
		X0 = i - t, //un-skewing the cell origin back to (x,y) space
		Y0 = j - t;
	dst_x[0] = x - X0; //The distance from the cell origin in (x,y) space
	dst_y[0] = y - Y0;

	//In 2D, simplex is just a triangle (equilateral)
	//Determine which simplex we are in
	//offsets for the middle corner of simplex in (i,j) space
	//lower triangle, XY order: (0,0)->(1,0)->(1,1)
	//otherwise upper triangle, YX order: (0,0)->(0,1)->(1,1)
	const unsigned int i1 = (dst_x[0] > dst_y[0]) ? 1u : 0u,
		j1 = 1u - i1;

	// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	// c = (3-sqrt(3))/6
	
	dst_x[1] = dst_x[0] - i1 + G2; //Now offset the middle corner in the un-skewed space (x,y)
	dst_y[1] = dst_y[0] - j1 + G2;
	dst_x[2] = dst_x[0] + H2; //Offset for the last corner in (x,y)
	dst_y[2] = dst_y[0] + H2;

	const unsigned char* const perm = permutation.Permutation;
	//Read the hashed gradient indices of the three simplex corner
	//logical AND of signed and unsigned will result in an unsigned number
	const unsigned int ii = i & 255u,
		jj = j & 255u;
	unsigned int grad_i[3];
	grad_i[0] = perm[ii + perm[jj]] % permutation.Gradient2DSize;
	grad_i[1] = perm[ii + i1 + perm[jj + j1]] % permutation.Gradient2DSize;
	grad_i[2] = perm[ii + 1u + perm[jj + 1u]] % permutation.Gradient2DSize;

	//calculate the weight from 3 corners
	for (unsigned int vertex = 0u; vertex < 3u; vertex++) {
		weight[vertex] = 0.5f - dst_x[vertex] * dst_x[vertex] - dst_y[vertex] * dst_y[vertex];
		if (weight[vertex] <= 0.0f) {
			corner[vertex] = 0.0f;
		} else {
			weight[vertex] *= weight[vertex];
			corner[vertex] = weight[vertex] * weight[vertex]
				* dot2D(grad2D(grad_i[vertex], 0u), grad2D(grad_i[vertex], 1u), dst_x[vertex], dst_y[vertex]);
		}
	}

	//add all the weights together
	//scale the result to [-1,1]
	return 70.0f * (corner[0] + corner[1] + corner[2]);
}

__device__ float STPSimplexNoise::simplex2DFractal(const STPPermutation& permutation,
	const float x, const float y, const STPFractalSimplexInformation& desc) {
	//extract all settings
	const auto [pers, lacu, oct, half_dim, offset, scale, init_amp, init_freq] = desc;
	float fractal = 0.0f;
	float amplitude = init_amp,
		frequency = init_freq;
	float absRange = 0.0f;

	//multiple phases of noise
	for (unsigned int i = 0u; i < oct; i++) {
		//subtract the half width and height can make the scaling focus at the centre
		const float sampleX = ((x - half_dim.x) + offset.x) / scale * frequency,
			//since the y is inverted we want to flip it over
			sampleY = ((y - half_dim.y) + offset.y) / scale * frequency;
		fractal += STPSimplexNoise::simplex2D(permutation, sampleX, sampleY) * amplitude;

		//accumulate range
		absRange += amplitude;
		//scale the parameters
		amplitude *= pers;
		frequency *= lacu;
	}

	//normalise result to [0.0, 1.0]
	return __saturatef((fractal + absRange) / (2.0f * absRange));
}