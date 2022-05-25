#include <SuperRealism+/Scene/STPSceneLight.h>
#include <SuperRealism+/Scene/Light/STPAmbientLight.h>
#include <SuperRealism+/Scene/Light/STPDirectionalLight.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLAD
#include <glad/glad.h>

using std::move;

using glm::vec3;

using namespace SuperTerrainPlus::STPRealism;

//STPSceneLight.h

STPSceneLight::STPSceneLight(STPLightSpectrum&& spectrum, STPLightType type) : Type(type), LightSpectrum(move(spectrum)) {

}

STPLightShadow* STPSceneLight::getLightShadow() {
	return const_cast<STPLightShadow*>(const_cast<const STPSceneLight*>(this)->getLightShadow());
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPSceneLight::lightDataAddress() const {
	return *this->LightDataAddress;
}

//STPAmbientLight.h

/**
 * @brief Ambient light data buffer packed according to GL_NV_shader_buffer_load alignment rule.
*/
struct STPPackedAmbientLightBuffer {
public:

	float Ka;
	//--------------- Frequent changing data
	float SpecCoord;
	//---------------
	GLuint64 AmbSpec;

};

STPAmbientLight::STPAmbientLight(STPLightSpectrum&& spectrum) : STPSceneLight(move(spectrum), STPLightType::Ambient) {
	const STPPackedAmbientLightBuffer ambBuf = {
		0.0f,
		0.0f,
		this->LightSpectrum.spectrumHandle()
	};
	//allocate memory for ambient light data buffer
	this->LightData.bufferStorageSubData(&ambBuf, sizeof(STPPackedAmbientLightBuffer), 
		GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	//get the buffer address
	this->LightDataAddress = STPBindlessBuffer(this->LightData, GL_READ_ONLY);

	//get the pointer to ambient light spectrum coordinate
	this->AmbSpecCoord = reinterpret_cast<float*>(this->LightData.mapBufferRange(offsetof(STPPackedAmbientLightBuffer, SpecCoord), sizeof(float), 
		GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT));
}

const STPLightShadow* STPAmbientLight::getLightShadow() const {
	//you know, ambient light never casts shadow.
	return nullptr;
}

void STPAmbientLight::setSpectrumCoordinate(float coord) {
	*this->AmbSpecCoord = coord;
}

void STPAmbientLight::setAmbient(const STPEnvironment::STPLightSetting::STPAmbientLightSetting& ambient) {
	if (!ambient.validate()) {
		throw STPException::STPInvalidEnvironment("The ambient setting fails to validate");
	}

	this->LightData.bufferSubData(&ambient.AmbientStrength, sizeof(float), offsetof(STPPackedAmbientLightBuffer, Ka));
}

//STPDirectionalLight.h

/**
 * @brief Directional light data buffer packed according to GL_NV_shader_buffer_load alignment rule
*/
struct STPPackedDirectionalLightBuffer {
public:

	//--------------
	vec3 Dir;
	float SpecCoord;
	//--------------
	float Kd, Ks;
	GLuint64 DirSpec;
	GLuint64EXT DirShadow;

};

STPDirectionalLight::STPDirectionalLight(STPDirectionalLightShadow&& dir_shadow, STPLightSpectrum&& spectrum) : 
	STPSceneLight(move(spectrum), STPLightType::Directional), Shadow(move(dir_shadow)) {
	const STPPackedDirectionalLightBuffer dirBuf = {
		vec3(0.0f),
		0.0f,
		0.0f, 0.0f,
		this->LightSpectrum.spectrumHandle(),
		this->Shadow ? this->Shadow->shadowDataAddress() : 0ull
	};
	this->LightData.bufferStorageSubData(&dirBuf, sizeof(STPPackedDirectionalLightBuffer), 
		GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	this->LightDataAddress = STPBindlessBuffer(this->LightData, GL_READ_ONLY);

	unsigned char* const mappedDirBuf = reinterpret_cast<unsigned char*>(this->LightData.mapBufferRange(offsetof(STPPackedDirectionalLightBuffer, Dir),
		sizeof(vec3) + sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT));
	//assign mapped pointers
	this->Dir = reinterpret_cast<vec3*>(mappedDirBuf);
	this->DirSpecCoord = reinterpret_cast<float*>(mappedDirBuf + sizeof(vec3));
}

const STPLightShadow* STPDirectionalLight::getLightShadow() const {
	return &this->Shadow.value();
}

void STPDirectionalLight::setSpectrumCoordinate(float coord) {
	*this->DirSpecCoord = coord;
}

void STPDirectionalLight::setDirectional(const STPEnvironment::STPLightSetting::STPDirectionalLightSetting& directional) {
	if (!directional.validate()) {
		throw STPException::STPInvalidEnvironment("The directional setting fails to validate");
	}

	const float diff_spec[2] = {
		directional.DiffuseStrength,
		directional.SpecularStrength
	};
	this->LightData.bufferSubData(diff_spec, sizeof(float) * 2u, offsetof(STPPackedDirectionalLightBuffer, Kd));
}

void STPDirectionalLight::setLightDirection(const vec3& dir) {
	*this->Dir = dir;

	if (this->Shadow) {
		//update CSM shadow if this directional light should cast a shadow.
		this->Shadow->setDirection(dir);
	}
}