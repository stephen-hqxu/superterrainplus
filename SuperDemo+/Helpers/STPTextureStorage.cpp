#include "STPTextureStorage.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <SuperTerrain+/Exception/STPIOException.h>

using namespace STPDemo;

using std::string;

using glm::ivec3;

void STPTextureStorage::STPTextureFreer::operator()(stbi_uc* const img) const {
	stbi_image_free(img);
}

STPTextureStorage::STPTextureStorage(const string& filename, const int comp) {
	stbi_uc* const texture = stbi_load(filename.c_str(), &this->Property.x, &this->Property.y, &this->Property.z, comp);
	if (texture == nullptr) {
		throw STP_IO_EXCEPTION_CREATE("Unable to load image file \'" + filename + '\'');
	}

	//manage this texture memory
	this->Texture = STPTextureMemmory(texture);
}

bool STPTextureStorage::empty() const noexcept {
	return !static_cast<bool>(this->Texture);
}

const ivec3& STPTextureStorage::property() const noexcept {
	return this->Property;
}

const stbi_uc* STPTextureStorage::texture() const noexcept {
	return this->Texture.get();
}