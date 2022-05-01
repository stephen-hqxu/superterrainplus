#include <SuperRealism+/Scene/STPMaterialLibrary.h>

//Exception
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

#include <limits>
#include <algorithm>

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

STPMaterialLibrary::STPMaterialLibrary(STPMaterialCount count) : MaterialCount(1u), MaxMaterialAllowance(count + 1u) {
	if (count > std::numeric_limits<STPMaterialID>::max()) {
		throw STPException::STPMemoryError("The number of element must be within the numeric range of material ID");
	}

	//allocate memory
	this->MaterialMemory = make_unique<STPMaterialProperty[]>(this->MaxMaterialAllowance);
	std::fill_n(this->MaterialMemory.get(), this->MaxMaterialAllowance, STPMaterialLibrary::DefaultMaterial);
	//send to device
	this->MaterialBuffer.bufferStorageSubData(this->MaterialMemory.get(), this->MaxMaterialAllowance * sizeof(STPMaterialProperty), GL_MAP_WRITE_BIT);
}

STPMaterialLibrary::STPMaterialID STPMaterialLibrary::add(const STPMaterialProperty& mat_prop) {
	if (this->MaterialCount >= this->MaxMaterialAllowance) {
		throw STPException::STPMemoryError("The number of material has reached the maximum material allowance");
	}

	//all materials are laid out in the memory in a contiguous manner
	//material ID serves as an index
	const STPMaterialID mat_idx = this->MaterialCount;

	this->MaterialMemory[mat_idx] = mat_prop;
	//send to device
	STPMaterialProperty* const mat_device = reinterpret_cast<STPMaterialProperty*>(
		this->MaterialBuffer.mapBufferRange(sizeof(STPMaterialProperty) * mat_idx, sizeof(STPMaterialProperty), GL_MAP_WRITE_BIT));
	*mat_device = mat_prop;
	this->MaterialBuffer.unmapBuffer();

	this->MaterialCount++;
	return mat_idx;
}

const STPMaterialLibrary::STPMaterialProperty& STPMaterialLibrary::operator[](STPMaterialID id) const {
	if (id > this->MaterialCount - 1u) {
		throw STPException::STPMemoryError("Material ID is invalid");
	}
	return this->MaterialMemory[id];
}

const STPBuffer& STPMaterialLibrary::operator*() const {
	return this->MaterialBuffer;
}