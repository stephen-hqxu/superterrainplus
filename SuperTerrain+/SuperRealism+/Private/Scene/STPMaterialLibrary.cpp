#include <SuperRealism+/Scene/STPMaterialLibrary.h>

//Exception
#include <SuperTerrain+/Exception/STPInsufficientMemory.h>

//GLAD
#include <glad/glad.h>

#include <limits>
#include <algorithm>

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

STPMaterialLibrary::STPMaterialLibrary(const STPMaterialCount count) : MaterialCount(1u), MaxMaterialAllowance(count + 1u) {
	//allocate memory
	this->MaterialMemory = make_unique<STPMaterialProperty[]>(this->MaxMaterialAllowance);
	std::fill_n(this->MaterialMemory.get(), this->MaxMaterialAllowance, STPMaterialLibrary::DefaultMaterial);
	//send to device
	this->MaterialBuffer.bufferStorageSubData(this->MaterialMemory.get(), this->MaxMaterialAllowance * sizeof(STPMaterialProperty), GL_MAP_WRITE_BIT);
}

STPMaterialLibrary::STPMaterialID STPMaterialLibrary::add(const STPMaterialProperty& mat_prop) {
	STP_ASSERTION_MEMORY_SUFFICIENCY(this->MaterialCount, 1u, this->MaxMaterialAllowance, "number of material");

	//all materials are laid out in the memory in a contiguous manner
	//material ID serves as an index
	const STPMaterialID mat_idx = this->MaterialCount;

	this->MaterialMemory[mat_idx] = mat_prop;
	//send to device
	STPMaterialProperty* const mat_device = new (this->MaterialBuffer.mapBufferRange(
		sizeof(STPMaterialProperty) * mat_idx, sizeof(STPMaterialProperty), GL_MAP_WRITE_BIT)) STPMaterialProperty;
	*mat_device = mat_prop;
	this->MaterialBuffer.unmapBuffer();

	this->MaterialCount++;
	return mat_idx;
}

STPMaterialLibrary::STPMaterialCount STPMaterialLibrary::size() const noexcept {
	return this->MaterialCount;
}

const STPMaterialLibrary::STPMaterialProperty& STPMaterialLibrary::operator[](const STPMaterialID id) const noexcept {
	return this->MaterialMemory[id];
}

const STPBuffer& STPMaterialLibrary::operator*() const noexcept {
	return this->MaterialBuffer;
}