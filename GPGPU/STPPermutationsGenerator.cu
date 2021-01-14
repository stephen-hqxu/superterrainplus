#include "STPPermutationsGenerator.cuh"

using namespace SuperTerrainPlus::STPCompute;

__host__ STPPermutationsGenerator::STPPermutationsGenerator(unsigned long long seed, unsigned int distribution, double offset) : GRADIENT2D_SIZE(distribution) {
	//seed the engine
	STPPermutationRNG rng;
	rng.seed(seed);

	//we allocate memory in cpu and shuffle the table, then copy back to gpu
	//I was thinking about unified memory but we don't need the memory on host after the init process
	//so using pure device memory will be faster to access than unified one
	//allocation
	int* PERMUTATIONS_HOST = new int[512];
	//copy one... copy first
	copy(begin(this->INIT_TABLE), end(this->INIT_TABLE), PERMUTATIONS_HOST);
	//shuffle first, the two copy must be the same
	shuffle(PERMUTATIONS_HOST, PERMUTATIONS_HOST + 256, rng);
	//copy this the shuffled result
	copy(PERMUTATIONS_HOST, PERMUTATIONS_HOST + 256, PERMUTATIONS_HOST + 256);
		
	//now copy the host table to the device
	cudaMalloc(&this->PERMUTATIONS, sizeof(int) * 512);
	cudaMemcpy(this->PERMUTATIONS, PERMUTATIONS_HOST, sizeof(int) * 512, cudaMemcpyHostToDevice);

	//finishing up
	delete[] PERMUTATIONS_HOST;

	//generate the gradient table
	//we are going to distribute the gradient evenly in a circle
	const double step = 360.0 / this->GRADIENT2D_SIZE * 1.0;//in degree
	double* GRADIENT2D_HOST = new double[this->GRADIENT2D_SIZE * 2];//2D so we *2
	int counter = 0;
	for (double angle = 0.0; angle < 360.0; angle += step) {//in degree
		GRADIENT2D_HOST[counter * 2] = cos(STPPermutationsGenerator::PI * (angle + offset) / 180.0);
		GRADIENT2D_HOST[counter * 2 + 1] = sin(STPPermutationsGenerator::PI * (angle + offset) / 180.0);

		counter++;
	}

	shuffle(GRADIENT2D_HOST, GRADIENT2D_HOST + this->GRADIENT2D_SIZE * 2, rng);
	//copy the host gradient to device
	cudaMalloc(&this->GRADIENT2D, sizeof(double) * this->GRADIENT2D_SIZE * 2);
	cudaMemcpy(this->GRADIENT2D, GRADIENT2D_HOST, sizeof(double) * this->GRADIENT2D_SIZE * 2, cudaMemcpyHostToDevice);
	//finishing up
	delete[] GRADIENT2D_HOST;
}

__host__ STPPermutationsGenerator::~STPPermutationsGenerator() {
	cudaFree(this->PERMUTATIONS);
	cudaFree(this->GRADIENT2D);
}

__device__ int STPPermutationsGenerator::perm(int index) {
	return this->PERMUTATIONS[index];//device memory can be accessed in device directly
}

__device__ double STPPermutationsGenerator::grad2D(int index, int component) {
	return this->GRADIENT2D[index * 2 + component];
}