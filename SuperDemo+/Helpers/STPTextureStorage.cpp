#include "STPTextureStorage.h"

//Export Implementation
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <stdexcept>

using namespace STPDemo;

using std::string;

using glm::ivec3;

void STPTextureStorage::STPTextureFreer::operator()(stbi_uc* img) const {
	stbi_image_free(img);
}

STPTextureStorage::STPTextureStorage(const string& filename, int comp) {
	this->Property = ivec3();
	stbi_uc* const texture = stbi_load(filename.c_str(), &this->Property.x, &this->Property.y, &this->Property.z, comp);
	if (texture == nullptr) {
		throw std::runtime_error("Unable to open file \'" + filename + "\'.");
	}

	//manage this texture memory
	this->Texture = STPTextureMemmory(texture);
}

bool STPTextureStorage::empty() const {
	return !static_cast<bool>(this->Texture);
}

const ivec3& STPTextureStorage::property() const {
	return this->Property;
}

const stbi_uc* STPTextureStorage::texture() const {
	return this->Texture.get();
}