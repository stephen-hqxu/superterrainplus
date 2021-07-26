#pragma once
#include <Helpers/STPTextureStorage.h>

using namespace SuperTerrainPlus;

STPTextureStorage::STPTextureStorage(const int width, const int height, const int channel, unsigned char* const texture)
	: Width(width), Height(height), Channel(channel), Texture(texture) {

}

STPTextureStorage::~STPTextureStorage() {
	stbi_image_free(this->Texture);
}

STPTextureStorage* STPTextureStorage::loadTexture(const char* filename, int desiredChannel) {
	int x, y, channel;
	unsigned char* data = stbi_load(filename, &x, &y, &channel, desiredChannel);

	return new STPTextureStorage(x, y, channel, data);
}