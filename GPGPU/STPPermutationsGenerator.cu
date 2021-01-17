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

__host__ STPPermutationsGenerator::STPPermutationsGenerator(const STPPermutationsGenerator& obj) : GRADIENT2D_SIZE(obj.GRADIENT2D_SIZE) {
	//deep copy gradient and permutation
	if (obj.GRADIENT2D != nullptr) {
		const size_t len = sizeof(double) * this->GRADIENT2D_SIZE * 2;
		cudaMalloc(&this->GRADIENT2D, len);
		cudaMemcpy(this->GRADIENT2D, obj.GRADIENT2D, len, cudaMemcpyDeviceToDevice);
	}
	else {
		this->GRADIENT2D = nullptr;
	}

	if (obj.PERMUTATIONS != nullptr) {
		const size_t len = sizeof(int) * 512;
		cudaMalloc(&this->PERMUTATIONS, len);
		cudaMemcpy(this->PERMUTATIONS, obj.PERMUTATIONS, len, cudaMemcpyDeviceToDevice);
	}
	else {
		this->PERMUTATIONS = nullptr;
	}
}

__host__ STPPermutationsGenerator::STPPermutationsGenerator(STPPermutationsGenerator&& obj) noexcept : GRADIENT2D_SIZE(std::exchange(obj.GRADIENT2D_SIZE, 0u)) {
	//move the ptr
	if (obj.GRADIENT2D != nullptr) {
		//steal the pointer for our good!
		this->GRADIENT2D = std::exchange(obj.GRADIENT2D, nullptr);
	}
	else {
		this->GRADIENT2D = nullptr;
	}
	if (obj.PERMUTATIONS != nullptr) {
		this->PERMUTATIONS = std::exchange(obj.PERMUTATIONS, nullptr);
	}
	else {
		this->PERMUTATIONS = nullptr;
	}
}

__host__ STPPermutationsGenerator::~STPPermutationsGenerator() {
	if (this->GRADIENT2D != nullptr) {
		cudaFree(this->GRADIENT2D);
		this->GRADIENT2D = nullptr;
	}
	if (this->PERMUTATIONS != nullptr) {
		cudaFree(this->PERMUTATIONS);
		this->PERMUTATIONS = nullptr;
	}
}

__host__ STPPermutationsGenerator& STPPermutationsGenerator::operator=(const STPPermutationsGenerator& obj) {
	//avoid duplicate copy
	if (this == &obj) {
		return *this;
	}

	if (obj.GRADIENT2D != nullptr) {
		const size_t len = sizeof(double) * this->GRADIENT2D_SIZE * 2;
		if (this->GRADIENT2D != nullptr) {
			//free previous memory
			cudaFree(this->GRADIENT2D);
		}

		cudaMalloc(&this->GRADIENT2D, len);
		cudaMemcpy(this->GRADIENT2D, obj.GRADIENT2D, len, cudaMemcpyDeviceToDevice);
	}
	else if (this->GRADIENT2D != nullptr) {
		//free previous memory and assign nullptr
		cudaFree(this->GRADIENT2D);
		this->GRADIENT2D = nullptr;
	}
	this->GRADIENT2D_SIZE = obj.GRADIENT2D_SIZE;

	if (obj.PERMUTATIONS != nullptr) {
		const size_t len = sizeof(int) * 512;
		if (this->PERMUTATIONS != nullptr) {
			//free previous memory
			cudaFree(this->PERMUTATIONS);
		}

		cudaMalloc(&this->PERMUTATIONS, len);
		cudaMemcpy(this->PERMUTATIONS, obj.PERMUTATIONS, len, cudaMemcpyDeviceToDevice);
	}
	else if (this->PERMUTATIONS != nullptr) {
		//free previous memory
		cudaFree(this->PERMUTATIONS);
		this->PERMUTATIONS = nullptr;
	}

	return *this;
}

__host__ STPPermutationsGenerator& STPPermutationsGenerator::operator=(STPPermutationsGenerator&& obj) noexcept {
	if (this == &obj) {
		return *this;
	}

	if (obj.GRADIENT2D != nullptr) {
		if (this->GRADIENT2D != nullptr) {
			cudaFree(this->GRADIENT2D);
		}
		this->GRADIENT2D = std::exchange(obj.GRADIENT2D, nullptr);
	}
	else if (this->GRADIENT2D != nullptr) {
		//free previous memory
		cudaFree(this->GRADIENT2D);
		this->GRADIENT2D = nullptr;
	}
	this->GRADIENT2D_SIZE = std::exchange(obj.GRADIENT2D_SIZE, 0u);

	if (obj.PERMUTATIONS != nullptr) {
		if (this->PERMUTATIONS != nullptr) {
			cudaFree(this->PERMUTATIONS);
		}
		this->PERMUTATIONS = std::exchange(obj.PERMUTATIONS, nullptr);
	}
	else if (this->PERMUTATIONS != nullptr) {
		cudaFree(this->PERMUTATIONS);
		this->PERMUTATIONS = nullptr;
	}

	return *this;
}

__device__ int STPPermutationsGenerator::perm(int index) {
	return this->PERMUTATIONS[index];//device memory can be accessed in device directly
}

__device__ double STPPermutationsGenerator::grad2D(int index, int component) {
	return this->GRADIENT2D[index * 2 + component];
}

__device__ unsigned int STPPermutationsGenerator::grad2D_size() {
	return this->GRADIENT2D_SIZE;
}