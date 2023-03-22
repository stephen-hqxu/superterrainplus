#include <SuperTerrain+/World/Diversity/Texture/STPTextureFactory.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
#include <SuperTerrain+/Exception/STPValidationFailed.h>

//GLAD
#include <glad/glad.h>

#include <algorithm>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::unique_ptr;
using std::make_unique;
using std::make_pair;
using std::move;
using std::numeric_limits;
using std::vector;

using glm::uvec2;

STPTextureFactory::STPTextureFactory(const STPTextureDatabase::STPDatabaseView& database_view,
	const STPEnvironment::STPChunkSetting& terrain_chunk, const STPSamplerStateModifier& modify_sampler) :
	MapDimension(terrain_chunk.MapSize), RenderDistance(terrain_chunk.RenderDistance),
	RenderDistanceCount(terrain_chunk.RenderDistance.x * terrain_chunk.RenderDistance.y),
	LocalChunkInfo(STPSmartDeviceMemory::makeDevice
		<STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation[]>(RenderDistanceCount)),
	ValidType(database_view.getValidMapType()) {
	//temporary cache
	STPIDConverter<STPTextureInformation::STPTextureID> textureID_converter;
	STPIDConverter<STPTextureType> textureType_converter;

	typedef STPTextureDatabase::STPDatabaseView DbView;
	//get all the data
	const DbView::STPMapGroupRecord group_rec = database_view.getValidMapGroup();
	const DbView::STPTextureRecord texture_rec = database_view.getValidTexture();
	const DbView::STPMapRecord texture_map_rec = database_view.getValidMap();
	const DbView::STPSampleRecord sample_rec = database_view.getValidSample();
	const DbView::STPAltitudeRecord altitude_rec = database_view.getAltitudes();
	const DbView::STPGradientRecord gradient_rec = database_view.getGradients();
	//checking if data are valid
	//sample not empty implies we have at least one splat rule of any
	STP_ASSERTION_VALIDATION(!group_rec.empty() && !texture_rec.empty() && !texture_map_rec.empty()
			&& !this->ValidType.empty() && !sample_rec.empty(), "Database contains empty thus invalid data and/or rules.");
	const size_t UsedTypeCount = this->ValidType.size();

	//we build the data structure that holds all texture in groups first
	{
		using std::generate;
		using std::transform;
		//we build the texture ID to index converter
		//loop through all texture collection
		textureID_converter.reserve(texture_rec.size());
		this->TextureViewRecord.reserve(texture_rec.size());
		for (auto [texture_it, texture_index] = make_pair(texture_rec.cbegin(), 0u); texture_it != texture_rec.cend(); texture_it++, texture_index++) {
			const auto& [texture_id, texture_view] = *texture_it;

			textureID_converter.emplace(texture_id, texture_index);
			this->TextureViewRecord.emplace_back(texture_view);
		}

		//build texture type converter
		//the purpose of the type converter is to eliminate unused type
		textureType_converter.reserve(UsedTypeCount);
		for (auto [type_it, type_index] = make_pair(this->ValidType.cbegin(), 0u); type_it != this->ValidType.cend(); type_it++, type_index++) {
			textureType_converter.emplace(*type_it, type_index);
		}

		//this is the total number of layered texture we will have
		const size_t textureCount = group_rec.size();
		this->Texture.resize(textureCount);
		this->TextureHandle.resize(textureCount);
		this->RawTextureHandle.resize(textureCount);
		//create texture
		generate(this->Texture.begin(), this->Texture.end(), []() {
			return STPSmartDeviceObject::makeGLTextureObject(GL_TEXTURE_2D_ARRAY);
		});

		//each texture ID contains some number of type as stride, if type is not use we set the index to
		this->TextureRegion.reserve(texture_map_rec.size());
		this->TextureRegionLookup.resize(database_view.Database.textureSize() * UsedTypeCount, STPTextureFactory::UnusedType);
		
		//loop through all texture data
		STPTextureInformation::STPMapGroupID prev_group = std::numeric_limits<STPTextureInformation::STPMapGroupID>::max();
		//group index wraps to 0 when incremented
		unsigned int layer_idx = 0u, group_idx = numeric_limits<unsigned int>::max();
		const STPTextureDatabase::STPMapGroupDescription* desc = nullptr;
		GLuint gl_texture = 0u;
		for (const auto [group_id, texture_id, type, img] : texture_map_rec) {
			//we know texture data has group index sorted in ascending order, the same as the group array
			//texture data is basically an "expanded" group array, they aligned in the same order
			if (prev_group != group_id) {
				//we meet a new group
				prev_group = group_id;
				group_idx++;
				layer_idx = 0u;

				//update group data
				const auto& [group_rec_id, member_count, group_props] = group_rec[group_idx];
				assert(group_rec_id == group_id && "Group ID from group record should be the same as that from texture map record.");
				desc = &group_props;
				gl_texture = this->Texture[group_idx].get();
				//allocate memory for each layer
				glTextureStorage3D(gl_texture, desc->MipMapLevel, desc->InteralFormat, desc->Dimension.x,
					desc->Dimension.y, static_cast<GLsizei>(member_count));
			}

			const unsigned int texture_idx = textureID_converter.at(texture_id);
			const auto type_idx = static_cast<STPTextureType_t>(textureType_converter.at(type));
			//populate memory for each layer
			glTextureSubImage3D(gl_texture, 0, 0, 0, layer_idx, desc->Dimension.x, desc->Dimension.y, 1,
				desc->ChannelFormat, desc->PixelFormat, img);

			//build data for renderer
			STPTextureInformation::STPTextureDataLocation& data_loc = this->TextureRegion.emplace_back(STPTextureInformation::STPTextureDataLocation());
			data_loc.GroupIndex = group_idx;
			data_loc.LayerIndex = layer_idx;
			this->TextureRegionLookup[texture_idx * UsedTypeCount + type_idx] =
				static_cast<unsigned int>(this->TextureRegion.size()) - 1u;

			//advance to the next layer in this group
			layer_idx++;
		}

		//create bindless texture handle
		transform(this->Texture.cbegin(), this->Texture.cend(), this->TextureHandle.begin(), [&modify_sampler](const auto& texture) {
			//change sampler parameter for the texture before creating bindless handle
			modify_sampler(texture.get());
			return STPSmartDeviceObject::makeGLBindlessTextureHandle(texture.get());
		});
		//and create a view to the texture handle so they can be passed to the renderer easier
		transform(this->TextureHandle.cbegin(), this->TextureHandle.cend(), this->RawTextureHandle.begin(),
			[](const auto& handle) { return handle.get(); });
	}

	//then we can start building splat rule data structure
	{
		vector<Sample> spalt_lookup;
		vector<STPTextureInformation::STPSplatRegistry> splat_reg;

		//loop through sample used
		spalt_lookup.reserve(sample_rec.size());
		splat_reg.reserve(sample_rec.size());
		//index counter
		unsigned int alt_acc = 0u, gra_acc = 0u;
		for (auto sample_it = sample_rec.cbegin(); sample_it != sample_rec.cend(); sample_it++) {
			//our sample is sorted in ascending order, and all splat tables are "expanded" version of sorted samples
			const auto [sample_id, alt_count, gra_count] = *sample_it;

			//build lookup table
			spalt_lookup.emplace_back(sample_id);
			STPTextureInformation::STPSplatRegistry& reg = splat_reg.emplace_back(STPTextureInformation::STPSplatRegistry());
			reg.AltitudeEntry = alt_acc;
			reg.AltitudeSize = static_cast<unsigned int>(alt_count);
			reg.GradientEntry = gra_acc;
			reg.GradientSize = static_cast<unsigned int>(gra_count);

			//update accumulator
			alt_acc += static_cast<unsigned int>(alt_count);
			gra_acc += static_cast<unsigned int>(gra_count);
		}

		//build the splat texture and replace texture ID with index to the texture database
		const vector<STPTextureInformation::STPAltitudeNode> alt_reg = STPTextureFactory::convertSplatID(altitude_rec, textureID_converter);
		const vector<STPTextureInformation::STPGradientNode> gra_reg = STPTextureFactory::convertSplatID(gradient_rec, textureID_converter);

		//copy those memory to device
		this->SplatLookup_d = move(STPTextureFactory::copyToDevice(spalt_lookup));
		this->SplatLookupCount = spalt_lookup.size();
		this->SplatRegistry_d = move(STPTextureFactory::copyToDevice(splat_reg));
		this->AltitudeRegistry_d = move(STPTextureFactory::copyToDevice(alt_reg));
		this->GradientRegistry_d = move(STPTextureFactory::copyToDevice(gra_reg));
	}
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

STPTextureInformation::STPSplatRuleDatabase STPTextureFactory::getSplatDatabase() const noexcept {
	return STPTextureInformation::STPSplatRuleDatabase{
		this->SplatLookup_d.get(),
		static_cast<unsigned int>(this->SplatLookupCount),
		this->SplatRegistry_d.get(),
		this->AltitudeRegistry_d.get(),
		this->GradientRegistry_d.get()
	};
}

template<typename T>
STPSmartDeviceMemory::STPDeviceMemory<T[]> STPTextureFactory::copyToDevice(const std::vector<T>& data) {
	STPSmartDeviceMemory::STPDeviceMemory<T[]> device = STPSmartDeviceMemory::makeDevice<T[]>(data.size());
	//copy to device
	STP_CHECK_CUDA(cudaMemcpy(device.get(), data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

	return device;
}

void STPTextureFactory::operator()(const cudaTextureObject_t biomemap_tex, const cudaTextureObject_t heightmap_tex,
	const cudaSurfaceObject_t splatmap_surf, const STPRequestingChunkInfo& requesting_local,
	const cudaStream_t stream) const {
	if (requesting_local.size() == 0u) {
		//nothing needs to be done
		return;
	}
	//too many rendered chunk than the memory we have allocation
	STP_ASSERTION_NUMERIC_DOMAIN(requesting_local.size() <= this->RenderDistanceCount,
		"The number of requesting local is more than the total number of rendered chunk.");

	//prepare the request
	STP_CHECK_CUDA(cudaMemcpyAsync(this->LocalChunkInfo.get(), requesting_local.data(), 
		requesting_local.size() * sizeof(STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation), cudaMemcpyHostToDevice, stream));
	//the LocalChunkID array is over-allocated, so we also need to send the size of actual data to device

	//launch
	this->splat(biomemap_tex, heightmap_tex, splatmap_surf, STPTextureInformation::STPSplatGeneratorInformation{
		this->LocalChunkInfo.get(),
		static_cast<unsigned int>(requesting_local.size())
	}, stream);
}

STPTextureInformation::STPSplatTextureDatabase STPTextureFactory::getSplatTexture() const noexcept {
	return STPTextureInformation::STPSplatTextureDatabase{
		this->RawTextureHandle.data(),
		this->RawTextureHandle.size(),

		this->TextureRegion.data(),
		this->TextureRegion.size(),

		this->TextureRegionLookup.data(),
		this->TextureRegionLookup.size(),

		this->TextureViewRecord.data(),
		this->TextureViewRecord.size()
	};
}

STPTextureFactory::STPTextureType_t STPTextureFactory::convertType(const STPTextureType type) const {
	auto type_beg = this->ValidType.cbegin();
	//check if type exists
	if (!std::binary_search(type_beg, this->ValidType.cend(), type)) {
		//type not found
		return STPTextureFactory::UnregisteredType;
	}

	//found
	const size_t type_index = std::lower_bound(type_beg, this->ValidType.cend(), type) - type_beg;
	return static_cast<STPTextureType_t>(type_index);
}

size_t STPTextureFactory::usedType() const noexcept {
	return this->ValidType.size();
}