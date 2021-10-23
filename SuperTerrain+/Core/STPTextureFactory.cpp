#include <SuperTerrain+/World/Diversity/Texture/STPTextureFactory.h>

//Import implementation
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

#include <algorithm>
#include <limits>

using namespace SuperTerrainPlus::STPDiversity;

using std::unique_ptr;
using std::make_unique;
using std::make_pair;
using std::make_tuple;
using std::numeric_limits;

using std::vector;

using glm::uvec2;

#define TEXTYPE_TYPE std::underlying_type_t<STPTextureType>
#define TEXTYPE_VALUE(TYPE) static_cast<TEXTYPE_TYPE>(TYPE)

STPTextureFactory::STPTextureFactory(const STPTextureDatabase::STPDatabaseView& database_view) {
	//we build the data structure that holds all texture in groups first
	{
		//get the data
		const STPTextureDatabase::STPDatabaseView::STPGroupRecord group = database_view.getValidGroup();
		const STPTextureDatabase::STPDatabaseView::STPTextureCollectionRecord collection = database_view.getValidTexture();
		const STPTextureDatabase::STPDatabaseView::STPTextureDataRecord data = database_view.getValidTextureData();

		//this is the total number of layered texture we will have
		this->Texture.resize(group.size());
		glCreateTextures(GL_TEXTURE_2D_ARRAY, this->Texture.size(), this->Texture.data());
		//loop through all groups
		//we can also iterate through the GL texture array at the same time since they have the same dimension
		this->GroupIDConverter.rehash(group.size());
		for (auto [group_it, gl_texture_it, group_index] = make_tuple(group.cbegin(), this->Texture.cbegin(), 0u);
			group_it != group.cend() && gl_texture_it != this->Texture.cend(); group_it++, gl_texture_it++, group_index++) {
			const auto [group_id, member_count, group_props] = *group_it;

			//allocate memory for each layer
			glTextureStorage3D(*gl_texture_it, 1, group_props->InteralFormat, group_props->Dimension.x, group_props->Dimension.y, member_count);

			//build group ID converter
			this->GroupIDConverter.emplace(group_id, group_index);
		}

		//now we build the texture ID to index converter
		//loop through all texture collection
		this->TextureIDConverter.rehash(collection.size());
		for (auto [texture_it, texture_index] = make_pair(collection.cbegin(), 0u); texture_it != collection.cend(); texture_it++, texture_index++) {
			this->TextureIDConverter.emplace(*texture_it, texture_index);
		}

		//each texture ID contains some number of type as stride, if type is not use we set the index to 
		this->TextureRegion.reserve(data.size());
		this->TextureRegionLookup.resize(database_view.Database.textureCollectionSize() * TEXTYPE_VALUE(STPTextureType::TypeCount), numeric_limits<unsigned int>::max());
		//loop through all texture data
		STPTextureInformation::STPTextureGroupID prev_group = numeric_limits<STPTextureInformation::STPTextureGroupID>::max();
		unsigned int layer_idx = 0u;
		for (const auto [group_id, texture_id, type, img] : data) {
			//we know texture data has group index sorted in ascending order, the same as the group array
			//texture data is basically an "expanded" group array, they aligned in the same order
			if (prev_group != group_id) {
				//we meet a new group, reset or update counter
				prev_group = group_id;
				layer_idx = 0u;
			}
			const unsigned int group_idx = this->GroupIDConverter[group_id],
				texture_idx = this->TextureIDConverter[texture_id];
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
		//get all data
		const STPTextureDatabase::STPDatabaseView::STPSampleRecord sample = database_view.getValidSample();
		const STPTextureDatabase::STPDatabaseView::STPAltitudeRecord altitude = database_view.getAltitudes();
		const STPTextureDatabase::STPDatabaseView::STPGradientRecord gradient = database_view.getGradients();

		//loop through sample used
		this->SplatLookup.reserve(sample.size());
		this->SplatRegistry.resize(sample.size());
		//index counter
		unsigned int alt_acc = 0u, gra_acc = 0u;
		for (auto [sample_it, sample_index] = make_pair(sample.cbegin(), 0u); sample_it != sample.cend(); sample_it++, sample_index++) {
			//our sample is sorted in asc order, and all splat tables are "expanded" version of sorted samples
			const auto [sample_id, alt_count, gra_count] = *sample_it;

			//build splat data table

			//build lookup table
			this->SplatLookup.emplace_back(sample_id);
			STPTextureInformation::STPSplatRegistry& reg = this->SplatRegistry.emplace_back(STPTextureInformation::STPSplatRegistry());
			reg.AltitudeEntry = alt_acc;
			reg.AltitudeSize = alt_count;
			reg.GradientEntry = gra_acc;
			reg.GradientSize = gra_count;

			//update accumulator
			alt_acc += alt_count;
			gra_acc += gra_count;
		}

		//build the splat texture and replace texture ID with index to the texture database
		this->convertSplatID(altitude, this->AltitudeRegistry);
		this->convertSplatID(gradient, this->GradientRegistry);

	}
}

STPTextureFactory::~STPTextureFactory() {
	//delete all gl texture
	glDeleteTextures(this->Texture.size(), this->Texture.data());
}

template<typename N>
void STPTextureFactory::convertSplatID(const STPTextureDatabase::STPDatabaseView::STPNodeRecord<N>& node, vector<N>& reg) {
	//loop through splat data
	reg.reserve(node.size());
	for (const auto& curr_node : node) {
		auto& id = reg.emplace_back(curr_node.second).Reference;
		id.RegionIndex = this->TextureIDConverter[id.DatabaseKey];
	}
}