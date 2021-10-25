#include <SuperTerrain+/World/Diversity/Texture/STPTextureFactory.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
//Import implementation
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

//GLAD
#include <glad/glad.h>

#include <algorithm>
#include <limits>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::unique_ptr;
using std::make_unique;
using std::make_pair;
using std::make_tuple;
using std::numeric_limits;
using std::move;

using std::vector;

using glm::uvec2;

#define TEXTYPE_TYPE std::underlying_type_t<STPTextureType>
#define TEXTYPE_VALUE(TYPE) static_cast<TEXTYPE_TYPE>(TYPE)

STPTextureFactory::STPTextureFactory(const STPTextureDatabase::STPDatabaseView& database_view) {
	//temporary cache
	STPIDConverter<STPTextureInformation::STPTextureID> textureID_converter;
	STPIDConverter<STPTextureInformation::STPTextureGroupID> groupID_converter;

	//we build the data structure that holds all texture in groups first
	{
		//get the data
		const STPTextureDatabase::STPDatabaseView::STPGroupRecord group = database_view.getValidGroup();
		const STPTextureDatabase::STPDatabaseView::STPTextureRecord texture = database_view.getValidTexture();
		const STPTextureDatabase::STPDatabaseView::STPTextureDataRecord texture_map = database_view.getValidMap();

		//this is the total number of layered texture we will have
		this->Texture.resize(group.size());
		glCreateTextures(GL_TEXTURE_2D_ARRAY, this->Texture.size(), this->Texture.data());
		//loop through all groups
		//we can also iterate through the GL texture array at the same time since they have the same dimension
		groupID_converter.rehash(group.size());
		for (auto [group_it, gl_texture_it, group_index] = make_tuple(group.cbegin(), this->Texture.cbegin(), 0u);
			group_it != group.cend() && gl_texture_it != this->Texture.cend(); group_it++, gl_texture_it++, group_index++) {
			const auto [group_id, member_count, group_props] = *group_it;

			//allocate memory for each layer
			glTextureStorage3D(*gl_texture_it, 1, group_props->InteralFormat, group_props->Dimension.x, group_props->Dimension.y, member_count);

			//build group ID converter
			groupID_converter.emplace(group_id, group_index);
		}

		//now we build the texture ID to index converter
		//loop through all texture collection
		textureID_converter.rehash(texture.size());
		for (auto [texture_it, texture_index] = make_pair(texture.cbegin(), 0u); texture_it != texture.cend(); texture_it++, texture_index++) {
			textureID_converter.emplace(*texture_it, texture_index);
		}

		//each texture ID contains some number of type as stride, if type is not use we set the index to 
		this->TextureRegion.reserve(texture_map.size());
		this->TextureRegionLookup.resize(database_view.Database.textureSize() * TEXTYPE_VALUE(STPTextureType::TypeCount), numeric_limits<unsigned int>::max());
		//loop through all texture data
		STPTextureInformation::STPTextureGroupID prev_group = numeric_limits<STPTextureInformation::STPTextureGroupID>::max();
		unsigned int layer_idx = 0u;
		for (const auto [group_id, texture_id, type, img] : texture_map) {
			//we know texture data has group index sorted in ascending order, the same as the group array
			//texture data is basically an "expanded" group array, they aligned in the same order
			if (prev_group != group_id) {
				//we meet a new group, reset or update counter
				prev_group = group_id;
				layer_idx = 0u;
			}
			const unsigned int group_idx = groupID_converter[group_id],
				texture_idx = textureID_converter[texture_id];
			const TEXTYPE_TYPE type_idx = TEXTYPE_VALUE(type);
			const STPTextureDatabase::STPTextureDescription* const desc = std::get<2>(group[group_idx]);
			const uvec2 dimension = desc->Dimension;

			//populate memory for each layer
			glTextureSubImage3D(this->Texture[group_idx], 0, 0, 0, layer_idx, dimension.x, dimension.y, 1, desc->ChannelFormat, desc->PixelFormat, img);

			//build data for renderer
			STPTextureInformation::STPTextureDataLocation& data_loc = this->TextureRegion.emplace_back(STPTextureInformation::STPTextureDataLocation());
			data_loc.GroupIndex = group_idx;
			data_loc.LayerIndex = layer_idx++;
			this->TextureRegionLookup[texture_idx * TEXTYPE_VALUE(STPTextureType::TypeCount) + type_idx] =
				static_cast<unsigned int>(this->TextureRegion.size()) - 1u;
		}
	}

	//then we can start building splat rule data structure
	{
		vector<Sample> spalt_lookup;
		vector<STPTextureInformation::STPSplatRegistry> splat_reg;
		//get all data
		const STPTextureDatabase::STPDatabaseView::STPSampleRecord sample = database_view.getValidSample();
		const STPTextureDatabase::STPDatabaseView::STPAltitudeRecord altitude = database_view.getAltitudes();
		const STPTextureDatabase::STPDatabaseView::STPGradientRecord gradient = database_view.getGradients();

		//loop through sample used
		spalt_lookup.reserve(sample.size());
		splat_reg.resize(sample.size());
		//index counter
		unsigned int alt_acc = 0u, gra_acc = 0u;
		for (auto [sample_it, sample_index] = make_pair(sample.cbegin(), 0u); sample_it != sample.cend(); sample_it++, sample_index++) {
			//our sample is sorted in asc order, and all splat tables are "expanded" version of sorted samples
			const auto [sample_id, alt_count, gra_count] = *sample_it;

			//build splat data table

			//build lookup table
			spalt_lookup.emplace_back(sample_id);
			STPTextureInformation::STPSplatRegistry& reg = splat_reg.emplace_back(STPTextureInformation::STPSplatRegistry());
			reg.AltitudeEntry = alt_acc;
			reg.AltitudeSize = alt_count;
			reg.GradientEntry = gra_acc;
			reg.GradientSize = gra_count;

			//update accumulator
			alt_acc += alt_count;
			gra_acc += gra_count;
		}

		//build the splat texture and replace texture ID with index to the texture database
		const vector<STPTextureInformation::STPAltitudeNode> alt_reg = STPTextureFactory::convertSplatID(altitude, textureID_converter);
		const vector<STPTextureInformation::STPGradientNode> gra_reg = STPTextureFactory::convertSplatID(gradient, textureID_converter);

		//copy those memory to device
		this->SplatLookup_d = move(STPTextureFactory::copyToDevice(spalt_lookup));
		this->SplatRegistry_d = move(STPTextureFactory::copyToDevice(splat_reg));
		this->AltitudeRegistry_d = move(STPTextureFactory::copyToDevice(alt_reg));
		this->GradientRegistry_d = move(STPTextureFactory::copyToDevice(gra_reg));
	}
}

STPTextureFactory::~STPTextureFactory() {
	//delete all gl texture
	glDeleteTextures(this->Texture.size(), this->Texture.data());
}

template<typename N>
vector<N> STPTextureFactory::convertSplatID 
	(const STPTextureDatabase::STPDatabaseView::STPNodeRecord<N>& node, const STPIDConverter<STPTextureInformation::STPTextureID>& converter) {
	vector<N> reg;
	//loop through splat data
	reg.reserve(node.size());
	for (const auto& curr_node : node) {
		auto& id = reg.emplace_back(curr_node.second).Reference;
		id.RegionIndex = converter.at(id.DatabaseKey);
	}

	return reg;
}

template<typename T>
STPSmartDeviceMemory::STPDeviceMemory<T[]> STPTextureFactory::copyToDevice(const std::vector<T>& data) {
	STPSmartDeviceMemory::STPDeviceMemory<T[]> device = STPSmartDeviceMemory::makeDevice<T[]>(data.size());
	//copy to device
	STPcudaCheckErr(cudaMemcpy(device.get(), data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

	return device;
}

STPTextureFactory::STPSplatDatabase STPTextureFactory::getSplatDatabase() const {
	return STPSplatDatabase{
		this->SplatLookup_d.get(),
		this->SplatRegistry_d.get(),
		this->AltitudeRegistry_d.get(),
		this->GradientRegistry_d.get()
	};
}