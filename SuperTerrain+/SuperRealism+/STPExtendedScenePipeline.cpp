#include <SuperRealism+/STPExtendedScenePipeline.h>
#include <SuperRealism+/STPRealismInfo.h>

//CUDA
#include <cuda_runtime.h>
//OptiX
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

//Error
#include <SuperRealism+/Utility/STPRendererErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPInsufficientMemory.h>
#include <SuperTerrain+/Exception/STPInvalidEnum.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
#include <SuperTerrain+/Exception/STPUnsupportedSystem.h>
#include <SuperTerrain+/Exception/STPValidationFailed.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>
#include <SuperRealism+/Utility/STPLogHandler.hpp>
//Engine
#include <SuperTerrain+/Utility/STPThreadPool.h>
#include <SuperTerrain+/GPGPU/STPDeviceRuntimeBinary.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//GL Object
#include <SuperRealism+/Object/STPBuffer.h>
//Shader
#include <SuperRealism+/Shader/RayTracing/STPScreenSpaceRayIntersection.cuh>
#include <SuperRealism+/Shader/RayTracing/STPInstanceIDCoder.cuh>

//CUDA-GL
#include <glad/glad.h>
#include <cuda_gl_interop.h>

//Container
#include <string>
#include <array>
#include <list>
#include <tuple>
#include <optional>
//Thread
#include <mutex>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstddef>

//GLM
#include <glm/common.hpp>
#include <glm/ext/matrix_transform.hpp>

using std::array;
using std::list;
using std::pair;
using std::tuple;
using std::string;
using std::string_view;
using std::optional;

using std::ostream;

using std::mutex;
using std::unique_lock;

using std::make_pair;
using std::make_optional;
using std::make_unique;
using std::to_string;
using std::for_each;

using glm::uvec2;
using glm::mat4;

using SuperTerrainPlus::STPUniqueResource;

using namespace SuperTerrainPlus::STPRealism;

//The preallocation size for log passed into OptiX compiler
constexpr static size_t mDefaultLogSize = 1024u;

inline static void loadExtendedShaderOption(const unsigned int arch, SuperTerrainPlus::STPDeviceRuntimeBinary::STPSourceInformation& info) {
	using SuperTerrainPlus::STPStringUtility::concatCharArray;
	//For performance consideration, always build with optimisation turned on;
	//we are not going to debug runtime compiled binary anyway.
	constexpr static auto optixIncludeOption = concatCharArray("-I ", STPRealismInfo::OptiXInclude);
	constexpr static auto optixSDKOption = concatCharArray("-I ", STPRealismInfo::OptiXSDK);

	info.Option
		["-arch=compute_" + to_string(arch)]
		["-std=c++17"]
		["-rdc=true"]
		["--use_fast_math"]
		["-default-device"]
		//include directory
		[optixIncludeOption.data()]
		[optixSDKOption.data()];
}

inline static void deviceContextDebugCallback(const unsigned int level, const char* const tag, const char* const message, void* const cbdata) {
	static constexpr auto getLevelStr = [](const unsigned int level) constexpr -> const char* {
		switch (level) {
		case 1u: return "FATAL";
		case 2u: return "ERROR";
		case 3u: return "WARNING";
		case 4u: return "PRINT";
		default: return "UNKNOWN";
		}
	};

	using std::endl;
	ostream& stream = *reinterpret_cast<ostream*>(cbdata);
	stream << "Level " << level << '(' << getLevelStr(level) << ")::" << tag << ':' << endl;
	stream << message << endl;
}

#define STP_OPTIX_SBT_HEADER_MEMBER __align__(OPTIX_SBT_RECORD_ALIGNMENT) unsigned char Header[OPTIX_SBT_RECORD_HEADER_SIZE]
/**
 * @brief The base shader binding record for packing.
 * @tparam T The data section of the shader record.
*/
template<class T>
struct STPSbtRecord {
public:

	using DataType = T;

	STP_OPTIX_SBT_HEADER_MEMBER;
	DataType Data;

	/**
	 * @brief Check to make sure the data field has no extra padding,
	 * to ensure `Header* + (Header Size) = Data*`.
	 * @return True if it satisfies the alignment requirement.
	*/
	inline constexpr static bool checkDataOffset() {
		return offsetof(STPSbtRecord, Data) == OPTIX_SBT_RECORD_HEADER_SIZE;
	}

};
//A specialisation denoting no data member.
template<>
struct STPSbtRecord<void> {
public:

	STP_OPTIX_SBT_HEADER_MEMBER;

};
#undef STP_OPTIX_SBT_HEADER_MEMBER

/**
 * @brief STPModuleDestroyer destroys OptiX module.
*/
struct STPModuleDestroyer {
public:

	inline void operator()(const OptixModule module) const {
		STP_CHECK_OPTIX(optixModuleDestroy(module));
	}

};
using STPSmartModule = STPUniqueResource<OptixModule, nullptr, STPModuleDestroyer>;
/**
 * @brief STPProgramGroupDestroyer destroys OptiX program group.
*/
struct STPProgramGroupDestroyer {
public:

	inline void operator()(const OptixProgramGroup pg) const {
		STP_CHECK_OPTIX(optixProgramGroupDestroy(pg));
	}

};
using STPSmartProgramGroup = STPUniqueResource<OptixProgramGroup, nullptr, STPProgramGroupDestroyer>;
/**
 * @brief STPPipelineDestroyer destroys OptiX pipeline.
*/
struct STPPipelineDestroyer {
public:

	inline void operator()(const OptixPipeline pipeline) const {
		STP_CHECK_OPTIX(optixPipelineDestroy(pipeline));
	}

};
using STPSmartPipeline = STPUniqueResource<OptixPipeline, nullptr, STPPipelineDestroyer>;

void STPExtendedScenePipeline::STPDeviceContextDestroyer::operator()(const OptixDeviceContext context) const {
	STP_CHECK_OPTIX(optixDeviceContextDestroy(context));
}

class STPExtendedScenePipeline::STPShaderMemoryInternal {
public:

	//The texture passed directly to the user
	STPTexture Texture;

private:

	//In case CUDA does not support GL format we use, need to pack the texture to this cache first and map to CUDA.
	//Then unpack back to the texture before passing to user.
	optional<STPBuffer> TextureCache;
	//Map to `TextureCache` if present, otherwise map to `Texture`.
	STPSmartDeviceObject::STPGraphicsResource TextureResource;

	uvec2 TextureResolution;

	/**
	 * @brief Initialisation parameter for a type of texture.
	 * Not all fields are used by every type.
	*/
	struct STPShaderMemoryDescription {
	public:

		//the copy direction of the memory, this is necessary only when texture cache is used.
		typedef unsigned char STPMemoryDirection;
		constexpr static STPMemoryDirection Input = 1u << 0u,
			Output = 1u << 1u;

		bool RequireTextureCache;
		//GL
		GLenum TextureInternal;
		GLenum TextureFormat;
		GLenum TexturePixel;
		//num. channel * sizeof(pixel type)
		size_t PixelSize;
		GLint CacheAlignment;
		//CUDA
		unsigned int ResourceRegisterFlag;
		bool NormalisedRead;

		STPMemoryDirection CopyDirection;

	};
	//pre-computed lookup table, using each type as index
	constexpr static auto MemoryTypeDescription = []() constexpr -> auto {
		array<STPShaderMemoryDescription, static_cast<STPShaderMemoryType_t>(STPShaderMemoryType::TotalTypeCount)> Desc = { };
		for (STPShaderMemoryType_t typeIdx = 0u; typeIdx < Desc.size(); typeIdx++) {
			STPShaderMemoryDescription& typeDesc = Desc[typeIdx];

			switch (const STPShaderMemoryType memType = static_cast<STPShaderMemoryType>(typeIdx)) {
			case STPShaderMemoryType::ScreenSpaceStencil:
				//although we said stencil is input / output, since CUDA cannot map stencil texture directly,
				//we need to pack it into the internal cache from an external texture, map the cache, then copy to the texture.
				typeDesc = {
					true,
					GL_STENCIL_INDEX8,
					GL_STENCIL_INDEX,
					GL_UNSIGNED_BYTE,
					sizeof(unsigned char),
					1,
					cudaGraphicsRegisterFlagsNone,
					false,
					STPShaderMemoryDescription::Output
				};
				break;
			case STPShaderMemoryType::ScreenSpaceRayDepth:
				typeDesc = {
					false,
					GL_R32F,
					GL_RED,
					GL_FLOAT,
					sizeof(float),
					4,
					cudaGraphicsRegisterFlagsReadOnly,
					false,
					STPShaderMemoryDescription::Input
				};
				break;
			case STPShaderMemoryType::ScreenSpaceRayDirection:
				typeDesc = {
					false,
					GL_RGBA16,
					GL_RGBA,
					GL_HALF_FLOAT,
					sizeof(unsigned short) * 4,
					8,
					cudaGraphicsRegisterFlagsReadOnly,
					true,
					STPShaderMemoryDescription::Input
				};
				break;
			case STPShaderMemoryType::GeometryPosition:
				typeDesc = {
					false,
					GL_RGBA32F,
					GL_RGBA,
					GL_FLOAT,
					sizeof(float) * 4,
					8,
					cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore,
					false,
					STPShaderMemoryDescription::Output
				};
				break;
			case STPShaderMemoryType::GeometryUV:
				typeDesc = {
					false,
					GL_RG16,
					GL_RG,
					GL_HALF_FLOAT,
					sizeof(unsigned short) * 2,
					4,
					cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore,
					true,
					STPShaderMemoryDescription::Output
				};
				break;
			default:
				throw STP_INVALID_ENUM_CREATE(memType, STPShaderMemoryType);
			}
		}

		return Desc;
	}();

	//description of texture and texture cache (if present)
	const STPShaderMemoryDescription TextureDesc;

	/**
	 * @brief Calculate the size of the texture cache.
	 * @return The size of texture cache.
	*/
	inline size_t calcBufferSize() const noexcept {
		return this->TextureResolution.x * this->TextureResolution.y * this->TextureDesc.PixelSize;
	}

public:

	const STPShaderMemoryType TextureType;

	/**
	 * @brief Initialise a new shader memory unit.
	 * @param type Specifies the type of shader memory.
	 * @param init_resolution Specifies the initial resolution.
	 * If any component is set to zero, no memory will be allocated and user is required to set the resolution manually later before using.
	 * However no error is generated in this case.
	*/
	STPShaderMemoryInternal(const STPShaderMemoryType type, const uvec2 init_resolution) :
		Texture(GL_TEXTURE_2D),
		TextureDesc(STPShaderMemoryInternal::MemoryTypeDescription[static_cast<STPShaderMemoryType_t>(type)]),
		TextureType(type) {
		if (init_resolution.x == 0u || init_resolution.y == 0u) {
			//invalid resolution, do not allocate any memory
			this->TextureResolution = init_resolution;
		} else {
			this->setResolution(init_resolution);
		}
	}

	STPShaderMemoryInternal(const STPShaderMemoryInternal&) = delete;

	STPShaderMemoryInternal(STPShaderMemoryInternal&&) = delete;

	STPShaderMemoryInternal& operator=(const STPShaderMemoryInternal&) = delete;

	STPShaderMemoryInternal& operator=(STPShaderMemoryInternal&&) = delete;

	~STPShaderMemoryInternal() = default;

	/**
	 * @brief Set the resolution of the shader memory.
	 * This will cause a memory reallocation.
	 * @param resolution The new resolution.
	*/
	void setResolution(const uvec2 resolution) {
		this->TextureResolution = resolution;
		
		//create new texture
		STPTexture texture(GL_TEXTURE_2D);
		texture.textureStorage2D(1, this->TextureDesc.TextureInternal, this->TextureResolution);

		using std::move;
		if (this->TextureDesc.RequireTextureCache) {
			STPBuffer texture_cache;
			texture_cache.bufferStorage(this->calcBufferSize(), GL_NONE);

			this->TextureResource = STPSmartDeviceObject::makeGLBufferResource(*texture_cache, this->TextureDesc.ResourceRegisterFlag);
			this->TextureCache = move(texture_cache);
		} else {
			this->TextureResource = STPSmartDeviceObject::makeGLImageResource(*texture, GL_TEXTURE_2D, this->TextureDesc.ResourceRegisterFlag);
		}
		this->Texture = move(texture);
	}

	/**
	 * @brief Copy a source texture into the internal texture cache.
	 * This function should only be called when the type of shader memory supports texture cache.
	 * Do not call this function while resource is being mapped, it is undefined behaviour.
	 * @param source The source texture to be copied.
	*/
	void copyFromTexture(const STPTexture& source) {
		this->TextureCache->bind(GL_PIXEL_PACK_BUFFER);
		//prepare for pixel transfer
		glPixelStorei(GL_PACK_ALIGNMENT, this->TextureDesc.CacheAlignment);
		source.getTextureImage(0, this->TextureDesc.TextureFormat, this->TextureDesc.TexturePixel, static_cast<STPOpenGL::STPsizei>(this->calcBufferSize()), 0);
	}

	/**
	 * @brief Map the memory so CUDA can safely use it.
	 * @tparam Mem_t Specifies the type of output of mapped memory.
	 * Currently only supports: a pointer type, STPSmartDeviceObject::STPTexture and STPSmartDeviceObject::STPSurface
	 * @param stream Specifies a CUDA stream.
	 * @return The mapped data, the exact type depends on the type of shader memory.
	*/
	template<class Mem_t>
	Mem_t mapMemory(const cudaStream_t stream) {
		if (this->TextureCache && ((this->TextureDesc.CopyDirection & STPShaderMemoryDescription::Input) != 0u)) {
			//copy from texture to cache
			this->copyFromTexture(this->Texture);
		}

		cudaGraphicsResource_t res = this->TextureResource.get();
		STP_CHECK_CUDA(cudaGraphicsMapResources(1, &res, stream));

		using std::is_same_v;
		using std::is_pointer_v;
		//get mapped data
		if constexpr (is_pointer_v<Mem_t>) {
			assert(this->TextureCache);
			//get mapped pointer from cache buffer
			void* mem;
			size_t memSize;
			STP_CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&mem, &memSize, res));
			assert(memSize == this->calcBufferSize());
			return reinterpret_cast<Mem_t>(mem);

			//usually an `else` after return is redundant, but since the return types are not the same, we want the compiler to discard unused branch.
		} else {
			//get mapped array from texture
			cudaResourceDesc memDesc = { };
			cudaArray_t mem;
			STP_CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&mem, res, 0u, 0u));
			memDesc.res.array.array = mem;
			memDesc.resType = cudaResourceTypeArray;

			if constexpr (is_same_v<Mem_t, STPSmartDeviceObject::STPSurface>) {
				assert((this->TextureDesc.CopyDirection & STPShaderMemoryDescription::Output) == 0u);
				//for output or IO type memory, always use surface
				return STPSmartDeviceObject::makeSurface(memDesc);
			} else {
				//for input type memory, a texture suffices
				cudaTextureDesc texDesc = { };
				texDesc.addressMode[0] = cudaAddressModeClamp;
				texDesc.addressMode[1] = cudaAddressModeClamp;
				texDesc.addressMode[2] = cudaAddressModeClamp;
				texDesc.readMode = this->TextureDesc.NormalisedRead ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
				return STPSmartDeviceObject::makeTexture(memDesc, texDesc);
			}
		}
	}

	/**
	 * @brief Unmap the memory and return control back to the renderer so CUDA cannot access it.
	 * @param stream Specifies a CUDA stream.
	*/
	void unmapMemory(const cudaStream_t stream) {
		cudaGraphicsResource_t res = this->TextureResource.get();
		STP_CHECK_CUDA(cudaGraphicsUnmapResources(1, &res, stream));

		if (this->TextureCache && ((this->TextureDesc.CopyDirection & STPShaderMemoryDescription::Output) != 0u)) {
			//copy from cache to texture
			this->TextureCache->bind(GL_PIXEL_UNPACK_BUFFER);
			glPixelStorei(GL_UNPACK_ALIGNMENT, this->TextureDesc.CacheAlignment);
			this->Texture.textureSubImage2D(0, STPGLVector::STPintVec2(0), this->TextureResolution,
				this->TextureDesc.TextureFormat, this->TextureDesc.TexturePixel, 0);
		}
	}

	/**
	 * @brief Clear up buffer binding state after finishing all operations in the rendering loop.
	*/
	inline static void finish() {
		STPBuffer::unbind(GL_PIXEL_PACK_BUFFER);
		STPBuffer::unbind(GL_PIXEL_UNPACK_BUFFER);
	}

};

class STPExtendedScenePipeline::STPMemoryManager {
private:

	const STPExtendedScenePipeline& Master;

	//All stuff below correspond to each traceable object lives in the scene graph.
	//Such that they all have the same length and same index.
	//These pointers remain unchanged once they are allocated to avoid any headache of updating and synchronisation.
	STPSmartDeviceMemory::STPDeviceMemory<OptixInstance[]> InstanceCache;
	STPSmartDeviceMemory::STPDeviceMemory<const STPGeometryAttributeFormat::STPVertexFormat* const*[]> PrimitiveGeometry;
	STPSmartDeviceMemory::STPDeviceMemory<const STPGeometryAttributeFormat::STPIndexFormat* const*[]> PrimitiveIndex;

	//A separate stream from the renderer, so that rendering and build task can overlap.
	STPSmartDeviceObject::STPStream ASBuildStream;
	//Record the event after the build operation, to make sure rendering does not start before TLAS build has finished.
	STPSmartDeviceObject::STPEvent ASBuildEvent;
	//The renderer doesn't need to allocate a lot of memory, so keep release threshold as default zero.
	STPSmartDeviceObject::STPMemPool BuildMemoryPool;
	
	typedef list<STPExtendedSceneObject::STPTraceable::STPGeometryUpdateInformation> STPUpdateList;
	//Record all traceable objects that requested for geometry update.
	//Once updates are finished, this record will be cleared.
	STPUpdateList PendingGeometryUpdate;
	mutable mutex UpdateListLock;

	STPSmartDeviceMemory::STPStreamedDeviceMemory<unsigned char[]> RootASMemory;
	OptixTraversableHandle RootASHandle;

public:

	//a storage of all user-created shader memory
	list<STPShaderMemoryInternal> ShaderMemoryStorage;

	//all AS build events should be submitted to this thread pool to ensure all asynchronous build tasks are finished before destruction
	STPThreadPool BuildWorker;

	const STPExtendedSceneObject::STPTraceable::STPGeometryUpdateNotifier ObjectUpdateNotifier =
		[this](const STPExtendedSceneObject::STPTraceable::STPGeometryUpdateInformation& geometry_info) -> void {
		unique_lock new_info_lock(this->UpdateListLock);
		//simply make a copy of the new info
		this->PendingGeometryUpdate.emplace_back(geometry_info);
	};

	/**
	 * @brief Initialise a STPMemoryManager instance.
	 * @param master The pointer to the master extended scene pipeline.
	*/
	STPMemoryManager(const STPExtendedScenePipeline& master) :
		Master(master), ASBuildStream(STPSmartDeviceObject::makeStream(cudaStreamNonBlocking)),
		ASBuildEvent(STPSmartDeviceObject::makeEvent(cudaEventDisableTiming)), RootASHandle{ }, BuildWorker(3u) {
		cudaMemPoolProps props = { };
		props.allocType = cudaMemAllocationTypePinned;
		props.handleTypes = cudaMemHandleTypeNone;
		props.location.type = cudaMemLocationTypeDevice;
		props.location.id = 0;
		this->BuildMemoryPool = STPSmartDeviceObject::makeMemPool(props);

		//allocate memory for scene objects
		const auto [lim_traceable] = this->Master.SceneMemoryLimit;
		this->InstanceCache = STPSmartDeviceMemory::makeDevice<OptixInstance[]>(lim_traceable);
		this->PrimitiveGeometry = STPSmartDeviceMemory::makeDevice<const STPGeometryAttributeFormat::STPVertexFormat* const*[]>(lim_traceable);
		this->PrimitiveIndex = STPSmartDeviceMemory::makeDevice<const STPGeometryAttributeFormat::STPIndexFormat* const*[]>(lim_traceable);
		//zero init all memory so we don't need to do anything when adding an new object to the scene.
		//Pointers can be remained NULL, because we are not using any of them so that's fine.
		//OptixInstance is a bit interesting:
		//- Traversable handle can be null to effectively *toggle off* the instance and AS build operation will ignore it.
		//- All the rests of the field can remain zero because this AS is totally ignored.
		STP_CHECK_CUDA(cudaMemset(this->InstanceCache.get(), 0x00, sizeof(OptixInstance) * lim_traceable));
		STP_CHECK_CUDA(cudaMemset(this->PrimitiveGeometry.get(), 0x00, sizeof(void*) * lim_traceable));
		STP_CHECK_CUDA(cudaMemset(this->PrimitiveIndex.get(), 0x00, sizeof(void*) * lim_traceable));
	}

	STPMemoryManager(const STPMemoryManager&) = delete;

	STPMemoryManager(STPMemoryManager&&) = delete;

	STPMemoryManager& operator=(const STPMemoryManager&) = delete;

	STPMemoryManager& operator=(STPMemoryManager&&) = delete;

	~STPMemoryManager() {
		this->syncWithASBuild();
	}

	/**
	 * @brief Get the pointers to the primitive data array.
	 * @return The geometry vertex and index respectively, and they are all on device memory.
	*/
	inline pair<const STPGeometryAttributeFormat::STPVertexFormat* const* const*,
		const STPGeometryAttributeFormat::STPIndexFormat* const* const*>
		getPrimitiveData() const noexcept {
		return make_pair(this->PrimitiveGeometry.get(), this->PrimitiveIndex.get());
	}

	/**
	 * @brief Check if there is update available to the root level AS.
	 * If so, initiate an asynchronous build event; otherwise return.
	 * This function does no synchronisation, but rendering should not begin until build has finished.
	 * @return The new AS handle to the root, if an update is initiated; otherwise nothing.
	 * This handle must not be used until build operation has finished.
	*/
	optional<OptixTraversableHandle> checkRootASUpdate() {
		STPUpdateList updatingObject;
		//check do we need to update
		{
			unique_lock check_update_lock(this->UpdateListLock);
			if (this->PendingGeometryUpdate.empty()) {
				//no update is needed
				return std::nullopt;
			}
			//move all pending updates to our internal memory
			updatingObject.splice(updatingObject.begin(), this->PendingGeometryUpdate);
		}
		//some commonly used data
		const size_t total_instance_count = this->Master.SceneMemoryCurrent.TraceableObject;
		const cudaStream_t build_stream = this->ASBuildStream.get();
		const cudaMemPool_t build_memPool = this->BuildMemoryPool.get();
		const OptixDeviceContext context = this->Master.Context.get();

		//update object data, we assume there is no rendering task using the object data and AS in progress
		//it is end-user's responsibility to call glFinish at the end of every frame, and ray tracing launches are initiated after checking for AS update
		OptixInstance* const instance_cache = this->InstanceCache.get();
		for_each(updatingObject.cbegin(), updatingObject.cend(),
			[instance_cache, prim_geometry = this->PrimitiveGeometry.get(), prim_index = this->PrimitiveIndex.get(), build_stream](const auto& update_info) {
			const auto& [traceable, update_instance, update_geometry, update_index] = update_info;
			//object ID is the same as index to the array of our objects
			const STPExtendedSceneObject::STPObjectID objID = traceable->TraceableObjectID;

			//copy new instance data to device code
			STP_CHECK_CUDA(cudaMemcpyAsync(instance_cache + objID, &update_instance, sizeof(OptixInstance), cudaMemcpyHostToDevice, build_stream));
			//update geometry data
			STP_CHECK_CUDA(cudaMemcpyAsync(prim_geometry + objID, update_geometry, sizeof(void*), cudaMemcpyHostToDevice, build_stream));
			STP_CHECK_CUDA(cudaMemcpyAsync(prim_index + objID, update_index, sizeof(void*), cudaMemcpyHostToDevice, build_stream));
			//swap object buffer, the swapped buffer should not be used until the build event has finished.
			traceable->swapBuffer();
		});
		//prepare build information
		OptixAccelBuildOptions tlas_options = { };
		tlas_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		tlas_options.operation = OPTIX_BUILD_OPERATION_BUILD;
		OptixBuildInput tlas_inputs = { };
		tlas_inputs.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		tlas_inputs.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_cache);
		tlas_inputs.instanceArray.numInstances = static_cast<unsigned int>(total_instance_count);

		//allocation for memory
		OptixAccelBufferSizes tlas_size = { };
		STP_CHECK_OPTIX(optixAccelComputeMemoryUsage(context, &tlas_options, &tlas_inputs, 1u, &tlas_size));
		//output, just return the old memory back to the pool, don't care about reusing
		this->RootASMemory = STPSmartDeviceMemory::makeStreamedDevice<unsigned char[]>(build_memPool, build_stream, tlas_size.outputSizeInBytes);
		//temp
		const STPSmartDeviceMemory::STPStreamedDeviceMemory<unsigned char[]> tlas_temp_mem =
			STPSmartDeviceMemory::makeStreamedDevice<unsigned char[]>(build_memPool, build_stream, tlas_size.tempSizeInBytes);

		//build TLAS
		STP_CHECK_OPTIX(optixAccelBuild(context, build_stream, &tlas_options, &tlas_inputs, 1u,
			reinterpret_cast<CUdeviceptr>(tlas_temp_mem.get()), tlas_size.tempSizeInBytes,
			reinterpret_cast<CUdeviceptr>(this->RootASMemory.get()), tlas_size.outputSizeInBytes,
			&this->RootASHandle, nullptr, 0u));
		//compact? idk, don't see it's worthy

		//record build event
		STP_CHECK_CUDA(cudaEventRecord(this->ASBuildEvent.get(), build_stream));

		return this->RootASHandle;
	}

	/**
	 * @brief Make a stream wait for an event recorded with build tasks.
	 * @param stream The stream to be waiting for AS build.
	*/
	inline void waitForASBuild(const cudaStream_t stream) const {
		STP_CHECK_CUDA(cudaStreamWaitEvent(stream, this->ASBuildEvent.get()));
	}

	/**
	 * @brief Make the current thread wait for AS build task to finish.
	*/
	inline void syncWithASBuild() const {
		STP_CHECK_CUDA(cudaStreamSynchronize(this->ASBuildStream.get()));
	}

};

class STPExtendedScenePipeline::STPScreenSpaceRayIntersection {
private:

	constexpr static auto SSRIShaderFilename =
		STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/RayTracing/STPScreenSpaceRayIntersection", ".cu");
	//shader record type
	typedef STPSbtRecord<void> STPLaunchedRayRecord;
	typedef STPSbtRecord<STPScreenSpaceRayIntersectionData::STPPrimitiveHitData> STPPrimitiveHitRecord;
	typedef STPSbtRecord<void> STPEnvironmentHitRecord;

	static_assert(STPPrimitiveHitRecord::checkDataOffset(),
		"SSRI shader binding table data has disallowed padding after the header.");

	const STPExtendedScenePipeline& Master;
	const OptixDeviceContext MasterContext;
	const cudaStream_t MasterStream;

	//the full pipeline containing the shader for ray intersection testing
	STPSmartPipeline IntersectionPipeline;
	array<unsigned int, 3u> IntersectionStackSize;

		//ray generation
	tuple<STPSmartDeviceMemory::STPDeviceMemory<STPLaunchedRayRecord>,
		//closest hit
		STPSmartDeviceMemory::STPDeviceMemory<STPPrimitiveHitRecord>,
		//miss
		STPSmartDeviceMemory::STPDeviceMemory<STPEnvironmentHitRecord>> IntersectionRecord;
	//the shader binding table for the intersection pipeline
	OptixShaderBindingTable IntersectionShader;

	const STPSmartDeviceMemory::STPDeviceMemory<STPScreenSpaceRayIntersectionData> IntersectionGlobalData;

	/**
	 * @brief Get a char pointer to the global data.
	 * @return A pointer to char of global data.
	*/
	inline unsigned char* getRawGlobalData() noexcept {
		return reinterpret_cast<unsigned char*>(this->IntersectionGlobalData.get());
	}

public:

	//Each primitive is assigned with a unique ID to identify them from the output texture during shading.
	//The environment ray is a special type of *primitive* indicating a missed ray, so it is reserved.
	//Each primitive should take one and only one ID from the rest of the available IDs.
	static constexpr size_t MaxPrimitiveRayID = STPScreenSpaceRayIntersectionData::EnvironmentRayID - 1u;

	/**
	 * @brief Initialise the STPScreenSpaceRayIntersection instance.
	 * @param master The pointer to the master extended scene pipeline.
	 * @param arch Specifies the CUDA device architecture for code generation.
	*/
	STPScreenSpaceRayIntersection(const STPExtendedScenePipeline& master, const unsigned int arch) :
		Master(master), MasterContext(this->Master.Context.get()),
		MasterStream(this->Master.RendererStream.get()), IntersectionShader{ },
		IntersectionGlobalData(STPSmartDeviceMemory::makeDevice<STPScreenSpaceRayIntersectionData>()) {
		/* ----------------------------------- compile into PTX --------------------------------- */
		STPDeviceRuntimeBinary::STPSourceInformation ssri_info;
		loadExtendedShaderOption(arch, ssri_info);
		ssri_info.NameExpression
			["SSRIData"]
			["__raygen__launchScreenSpaceRay"]
			["__closesthit__recordPrimitiveIntersection"]
			["__miss__recordEnvironmentIntersection"];
		//handle compilation output
		auto [ssri_program_object, ssri_log, ssri_expr] =
			STPDeviceRuntimeBinary::compile("STPScreenSpaceRayIntersection.cu", STPFile::read(SSRIShaderFilename.data()), ssri_info);
		const nvrtcProgram ssri_program = ssri_program_object.Program.get();
		STPLogHandler::handle(ssri_log);
		
		//prepare for log cache
		char log[mDefaultLogSize] = { };
		size_t logSize = mDefaultLogSize;

		OptixModule ssri_module;
		OptixPipelineCompileOptions ssri_pipeline_option = { };
		/* ------------------------------------ create module ----------------------------------- */
		{
			OptixModuleCompileOptions ssri_module_option = { };
			ssri_module_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			ssri_module_option.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;

			ssri_pipeline_option.numPayloadValues = 4;
			ssri_pipeline_option.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
			ssri_pipeline_option.pipelineLaunchParamsVariableName = ssri_expr.at("SSRIData").c_str();
			ssri_pipeline_option.usesPrimitiveTypeFlags =
				static_cast<decltype(OptixPipelineCompileOptions::usesPrimitiveTypeFlags)>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

			const auto [ssri_ptx, ssri_ptxSize] = STPDeviceRuntimeBinary::readPTX(ssri_program);

			STP_CHECK_OPTIX(optixModuleCreateFromPTX(this->MasterContext, &ssri_module_option,
				&ssri_pipeline_option, ssri_ptx.get(), ssri_ptxSize, log, &logSize, &ssri_module));
			//We don't really care if the actual log size is larger than the default allocated size,
			//overflown log will be abandoned.
		}
		const STPSmartModule ssri_module_manager(ssri_module);
		//logging
		STPLogHandler::handle(string_view(log, glm::min<size_t>(logSize, mDefaultLogSize)));
		//reset initial log size counter
		logSize = mDefaultLogSize;

		array<OptixProgramGroup, 3u> ssri_program_group = { };
		/* --------------------------------- create program group --------------------------------- */
		{
			const OptixProgramGroupOptions ssri_pg_option = { };
			OptixProgramGroupDesc ssri_pg_desc[3] = { };

			ssri_pg_desc[0] = { };
			ssri_pg_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			ssri_pg_desc[0].raygen.module = ssri_module;
			ssri_pg_desc[0].raygen.entryFunctionName = ssri_expr.at("__raygen__launchScreenSpaceRay").c_str();

			ssri_pg_desc[1] = { };
			ssri_pg_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			ssri_pg_desc[1].hitgroup.moduleCH = ssri_module;
			ssri_pg_desc[1].hitgroup.entryFunctionNameCH =
				ssri_expr.at("__closesthit__recordPrimitiveIntersection").c_str();

			ssri_pg_desc[2] = { };
			ssri_pg_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			ssri_pg_desc[2].miss.module = ssri_module;
			ssri_pg_desc[2].miss.entryFunctionName = ssri_expr.at("__miss__recordEnvironmentIntersection").c_str();

			STP_CHECK_OPTIX(optixProgramGroupCreate(this->MasterContext, ssri_pg_desc, 3u, &ssri_pg_option,
				log, &logSize, ssri_program_group.data()));
		}
		const array<STPSmartProgramGroup, ssri_program_group.size()> ssri_pg_manager = {
			STPSmartProgramGroup(ssri_program_group[0]),
			STPSmartProgramGroup(ssri_program_group[1]),
			STPSmartProgramGroup(ssri_program_group[2])
		};
		STPLogHandler::handle(string_view(log, glm::min<size_t>(logSize, mDefaultLogSize)));
		logSize = mDefaultLogSize;

		OptixPipeline ssri_pipeline;
		/* ----------------------------------- create pipeline --------------------------------------- */
		{
			OptixPipelineLinkOptions ssri_pipeline_link_option = { };
			ssri_pipeline_link_option.maxTraceDepth = 1u;
			ssri_pipeline_link_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

			STP_CHECK_OPTIX(optixPipelineCreate(this->MasterContext, &ssri_pipeline_option, &ssri_pipeline_link_option,
				ssri_program_group.data(), static_cast<unsigned int>(ssri_program_group.size()), log, &logSize, &ssri_pipeline));
			//store the pipeline, all previously used data can be deleted, and will be done automatically
			this->IntersectionPipeline = STPSmartPipeline(ssri_pipeline);

			//stack size configuration
			OptixStackSizes ssri_stack = { };
			for (const auto& pg : ssri_program_group) {
				STP_CHECK_OPTIX(optixUtilAccumulateStackSizes(pg, &ssri_stack));
			}
			auto& [traversal_stack, state_stack, continuation_stack] = this->IntersectionStackSize;
			STP_CHECK_OPTIX(optixUtilComputeStackSizes(&ssri_stack, 1u, 0u, 0u, &traversal_stack, &state_stack, &continuation_stack));
		}
		STPLogHandler::handle(string_view(log, glm::min<size_t>(logSize, mDefaultLogSize)));

		/* ---------------------------- allocate shader binding table -------------------------------- */
		{
			const auto [ch_vertex, ch_index] = this->Master.SceneMemory->getPrimitiveData();
			auto& [ssri_rec_rg, ssri_rec_ch, ssri_rec_ms] = this->IntersectionRecord;
			ssri_rec_rg = STPSmartDeviceMemory::makeDevice<STPLaunchedRayRecord>();
			ssri_rec_ch = STPSmartDeviceMemory::makeDevice<STPPrimitiveHitRecord>();
			ssri_rec_ms = STPSmartDeviceMemory::makeDevice<STPEnvironmentHitRecord>();
			STPLaunchedRayRecord rec_rg = { };
			STPPrimitiveHitRecord rec_ch = { };
			STPEnvironmentHitRecord rec_ms = { };
			//header packing
			STP_CHECK_OPTIX(optixSbtRecordPackHeader(ssri_program_group[0], &rec_rg));
			STP_CHECK_OPTIX(optixSbtRecordPackHeader(ssri_program_group[1], &rec_ch));
			STP_CHECK_OPTIX(optixSbtRecordPackHeader(ssri_program_group[2], &rec_ms));
			//load in initial data
			rec_ch.Data = { ch_vertex, ch_index };
			//record transfer
			STP_CHECK_CUDA(cudaMemcpy(ssri_rec_rg.get(), &rec_rg, sizeof(rec_rg), cudaMemcpyHostToDevice));
			STP_CHECK_CUDA(cudaMemcpy(ssri_rec_ch.get(), &rec_ch, sizeof(rec_ch), cudaMemcpyHostToDevice));
			STP_CHECK_CUDA(cudaMemcpy(ssri_rec_ms.get(), &rec_ms, sizeof(rec_ms), cudaMemcpyHostToDevice));

			//header assignment
			this->IntersectionShader.raygenRecord = reinterpret_cast<CUdeviceptr>(ssri_rec_rg.get());
			this->IntersectionShader.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(ssri_rec_ch.get());
			this->IntersectionShader.hitgroupRecordCount = 1u;
			this->IntersectionShader.hitgroupRecordStrideInBytes = sizeof(STPPrimitiveHitRecord);
			this->IntersectionShader.missRecordBase = reinterpret_cast<CUdeviceptr>(ssri_rec_ms.get());
			this->IntersectionShader.missRecordCount = 1u;
			this->IntersectionShader.missRecordStrideInBytes = sizeof(STPEnvironmentHitRecord);
		}

		/* ----------------------------------- global launch data ------------------------------------ */
		STPScreenSpaceRayIntersectionData ssri_data = { };
		constexpr mat4 init_invpv = glm::identity<mat4>();
		static_assert(sizeof(mat4) == sizeof(STPScreenSpaceRayIntersectionData::InvProjectionView),
			"Matrix type used in SSRI shader is incompatible with GLM matrix");
		memcpy(&ssri_data.InvProjectionView, &init_invpv, sizeof(mat4));

		STP_CHECK_CUDA(cudaMemcpy(this->IntersectionGlobalData.get(), &ssri_data, sizeof(STPScreenSpaceRayIntersectionData), cudaMemcpyHostToDevice));
	}

	STPScreenSpaceRayIntersection(const STPScreenSpaceRayIntersection&) = delete;

	STPScreenSpaceRayIntersection(STPScreenSpaceRayIntersection&&) = delete;

	STPScreenSpaceRayIntersection& operator=(const STPScreenSpaceRayIntersection&) = delete;

	STPScreenSpaceRayIntersection& operator=(STPScreenSpaceRayIntersection&&) = delete;

	~STPScreenSpaceRayIntersection() = default;

	/**
	 * @brief Update the stack size for intersection tracer pipeline.
	 * @param traversableDepth The maximum possible traversable graph depth for all traceable objects.
	*/
	inline void updateStackSize(const unsigned int traversableDepth) {
		const auto [traversal_stack, state_stack, continuation_stack] = this->IntersectionStackSize;
		//add one to the total depth because we have a top level IAS enclosing all other AS's.
		STP_CHECK_OPTIX(optixPipelineSetStackSize(this->IntersectionPipeline.get(), traversal_stack, state_stack, continuation_stack, traversableDepth + 1u));
	}

	/**
	 * @brief Update the internal record of traversable handle.
	 * @param handle The new handle to be used.
	*/
	inline void updateTraversableHandle(const OptixTraversableHandle handle) {
		STP_CHECK_CUDA(cudaMemcpyAsync(this->getRawGlobalData() + offsetof(STPScreenSpaceRayIntersectionData, Handle), &handle,
			sizeof(OptixTraversableHandle), cudaMemcpyHostToDevice, this->MasterStream));
	}

	/**
	 * @brief Update the inverse projection view matrix.
	 * @param inv_pv The pointer to the new inverse projection view matrix.
	*/
	inline void updateInvPV(const mat4& inv_pv) {
		STP_CHECK_CUDA(cudaMemcpyAsync(this->getRawGlobalData() + offsetof(STPScreenSpaceRayIntersectionData, InvProjectionView), &inv_pv,
			sizeof(mat4), cudaMemcpyHostToDevice, this->MasterStream));
	}

	/**
	 * @brief Launch the intersection program.
	 * @param textureData The texture data to be used during rendering.
	*/
	inline void launchIntersection(const STPScreenSpaceRayIntersectionData::STPTextureData& textureData) {
		//update texture data
		STP_CHECK_CUDA(cudaMemcpyAsync(this->getRawGlobalData() + offsetof(STPScreenSpaceRayIntersectionData, SSTexture), &textureData,
			sizeof(STPScreenSpaceRayIntersectionData::STPTextureData), cudaMemcpyHostToDevice, this->MasterStream));

		//mAgIc!
		const uvec2 resolution = this->Master.RenderResolution;
		STP_CHECK_OPTIX(optixLaunch(this->IntersectionPipeline.get(), this->MasterStream,
			reinterpret_cast<CUdeviceptr>(this->IntersectionGlobalData.get()), sizeof(STPScreenSpaceRayIntersectionData), &this->IntersectionShader,
			resolution.x, resolution.y, 1u
		));
	}

};

STPExtendedScenePipeline::STPShaderMemory::STPShaderMemory(STPShaderMemoryInternal* const internal) : Internal(internal) {
	
}

inline STPExtendedScenePipeline::STPShaderMemoryInternal* STPExtendedScenePipeline::STPShaderMemory::operator->() const {
	return this->Internal;
}

STPTexture& STPExtendedScenePipeline::STPShaderMemory::texture() noexcept {
	return this->Internal->Texture;
}

STPExtendedScenePipeline::STPShaderMemoryType STPExtendedScenePipeline::STPShaderMemory::type() const noexcept {
	return this->Internal->TextureType;
}

template<class Desc>
STPExtendedScenePipeline::STPValidatedInformation<Desc>::STPValidatedInformation(const Desc descriptor) : Description(descriptor) {

}

#define VALIDATE_SSRI_MEMORY(EXPR) STP_ASSERTION_VALIDATION(EXPR, "One or more specified shader memory have incompatible type with SSRI shader")

STPExtendedScenePipeline::STPValidatedInformation<STPExtendedScenePipeline::STPSSRIDescription>
	STPExtendedScenePipeline::STPSSRIDescription::validate() const {
	VALIDATE_SSRI_MEMORY(this->Stencil.type() == STPShaderMemoryType::ScreenSpaceStencil);
	VALIDATE_SSRI_MEMORY(this->RayDepth.type() == STPShaderMemoryType::ScreenSpaceRayDepth);
	VALIDATE_SSRI_MEMORY(this->RayDirection.type() == STPShaderMemoryType::ScreenSpaceRayDirection);
	VALIDATE_SSRI_MEMORY(this->Position.type() == STPShaderMemoryType::GeometryPosition);
	VALIDATE_SSRI_MEMORY(this->UV.type() == STPShaderMemoryType::GeometryUV);

	return STPValidatedInformation(*this);
}

inline static OptixDeviceContext createDeviceContext(const STPExtendedScenePipeline::STPScenePipelineInitialiser& init, ostream& stream) {
	OptixDeviceContextOptions ctx_option = { };
	ctx_option.logCallbackFunction = &deviceContextDebugCallback;
	ctx_option.logCallbackData = &stream;
	ctx_option.logCallbackLevel = init.LogLevel;
	ctx_option.validationMode = init.UseDebugContext ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;

	OptixDeviceContext dev_ctx;
	STP_CHECK_OPTIX(optixDeviceContextCreate(init.DeviceContext, &ctx_option, &dev_ctx));
	return dev_ctx;
}

STPExtendedScenePipeline::STPExtendedScenePipeline(const STPScenePipelineInitialiser& scene_init, ostream& msg_stream) :
	Context(createDeviceContext(scene_init, msg_stream)),
	RendererStream(STPSmartDeviceObject::makeStream(cudaStreamNonBlocking)), SceneMemoryCurrent{ }, SceneMemoryLimit(scene_init.ObjectCapacity),
	SceneMemory(make_unique<STPMemoryManager>(*this)), IntersectionTracer(make_unique<STPScreenSpaceRayIntersection>(*this, scene_init.TargetDeviceArchitecture)),
	RenderResolution(uvec2(0u)) {
	//traceable object max count check, which should be less than the bit width of standard stencil buffer
	const auto [object_max] = this->SceneMemoryLimit;
	STP_ASSERTION_NUMERIC_DOMAIN(object_max <= STPScreenSpaceRayIntersection::MaxPrimitiveRayID,
		"The extended scene memory limit should not exceed the allowance defined by each shader");

	//context check
	const OptixDeviceContext dev_ctx = this->Context.get();
	unsigned int maxInstanceID;
	STP_CHECK_OPTIX(optixDeviceContextGetProperty(dev_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &maxInstanceID, sizeof(unsigned int)));
	if (maxInstanceID < STPInstanceIDCoder::UserIDMask) {
		throw STP_UNSUPPORTED_SYSTEM_CREATE("The current OptiX rendering device does not support enough bits for IAS user ID");
	}
}

STPExtendedScenePipeline::~STPExtendedScenePipeline() {
	this->SceneMemory->syncWithASBuild();
	STP_CHECK_CUDA(cudaStreamSynchronize(this->RendererStream.get()));

	//kill all object dependencies
	//pending works on each object will still be running on the thread pool, and thread pool won't be dead until all tasks are finished.
	auto& object = this->SceneComponent.TraceableObjectDatabase;
	for_each(object.begin(), object.end(), [](auto traceable) {
		traceable->TraceableObjectID = STPExtendedSceneObject::EmptyObjectID;
		traceable->setSceneInformation(STPExtendedSceneObject::STPTraceable::STPSceneInformation { });
	});
}

void STPExtendedScenePipeline::updateAccelerationStructure() {
	const optional<OptixTraversableHandle> maybe_handle = this->SceneMemory->checkRootASUpdate();
	if (!maybe_handle) {
		//no update is needed
		return;
	}

	const OptixTraversableHandle handle = *maybe_handle;
	//send the new handle to each shader
	//the build operation may still be in progress, hence we should not launch the shader before any synchronisation happens.
	this->IntersectionTracer->updateTraversableHandle(handle);
}

void STPExtendedScenePipeline::add(STPExtendedSceneObject::STPTraceable& object) {
	size_t& curr_traceable = this->SceneMemoryCurrent.TraceableObject;
	STP_ASSERTION_MEMORY_SUFFICIENCY(curr_traceable, 1u, this->SceneMemoryLimit.TraceableObject, "The number of traceable object has reached the limit");
	auto& [object_db, object_depth] = this->SceneComponent;
	//record the old maximum traversable depth
	const unsigned int traversableDepth_old = object_depth.empty() ? 0u : *object_depth.crbegin();
	
	//add new object an associate it with dependency of the current scene pipeline
	object_db.emplace_back(&object);
	object_depth.emplace(object.traversableDepth());
	//the old size is the index of the newly inserted object
	object.TraceableObjectID = static_cast<STPExtendedSceneObject::STPObjectID>(curr_traceable);
	object.setSceneInformation(STPExtendedSceneObject::STPTraceable::STPSceneInformation {
		this->Context.get(),
		&this->SceneMemory->BuildWorker,
		this->SceneMemory->ObjectUpdateNotifier
	});

	//check if the max depth has changed
	const unsigned int traversableDepth_new = *object_depth.crbegin();
	if (traversableDepth_new > traversableDepth_old) {
		//need to increase stack size
		this->IntersectionTracer->updateStackSize(traversableDepth_new);
	}

	curr_traceable = object_db.size();
}

void STPExtendedScenePipeline::setResolution(const uvec2 resolution) {
	STP_ASSERTION_NUMERIC_DOMAIN(resolution.x > 0u && resolution.y > 0u, "Both components of render resolution must be positive");
	this->RenderResolution = resolution;

	auto& smStorage = this->SceneMemory->ShaderMemoryStorage;
	for_each(smStorage.begin(), smStorage.end(), [resolution](auto& internal) { internal.setResolution(resolution); });
}

void STPExtendedScenePipeline::setInverseProjectionView(const mat4& inv_pv) {
	this->IntersectionTracer->updateInvPV(inv_pv);
}

STPExtendedScenePipeline::STPShaderMemory STPExtendedScenePipeline::createShaderMemory(const STPShaderMemoryType type) {
	return &this->SceneMemory->ShaderMemoryStorage.emplace_back(type, this->RenderResolution);
}

void STPExtendedScenePipeline::destroyShaderMemory(const STPShaderMemory sm) {
	auto& smStorage = this->SceneMemory->ShaderMemoryStorage;
	smStorage.erase(std::find_if(smStorage.cbegin(), smStorage.cend(),
		[sm_ptr = const_cast<const STPShaderMemoryInternal*>(sm.Internal)](const auto& internal) { return &internal == sm_ptr; }));
}

void STPExtendedScenePipeline::traceIntersection(const STPTexture& stencil_in, const STPValidatedInformation<STPSSRIDescription>& ssri_info) {
	const cudaStream_t stream = this->RendererStream.get();
	const auto [stencil, ray_depth, ray_direction, position, tex_coord] = ssri_info.Description;
	//copy input stencil into the internal cache
	stencil->copyFromTexture(stencil_in);
	//map all of them and create texture and surface
	{
		const auto stencil_mem = stencil->mapMemory<unsigned char*>(stream);
		const auto ray_depth_mem = ray_depth->mapMemory<STPSmartDeviceObject::STPTexture>(stream);
		const auto ray_direction_mem = ray_direction->mapMemory<STPSmartDeviceObject::STPTexture>(stream);
		const auto position_mem = position->mapMemory<STPSmartDeviceObject::STPSurface>(stream);
		const auto tex_coord_mem = tex_coord->mapMemory<STPSmartDeviceObject::STPSurface>(stream);

		const STPScreenSpaceRayIntersectionData::STPTextureData texture_mem = {
			stencil_mem, ray_depth_mem.get(), ray_direction_mem.get(), position_mem.get(), tex_coord_mem.get()
		};

		//initiate intersection tracer
		this->SceneMemory->waitForASBuild(stream);
		this->IntersectionTracer->launchIntersection(texture_mem);

		//clean up, memory will be freed automatically
		STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	}
	stencil->unmapMemory(stream);
	ray_depth->unmapMemory(stream);
	ray_direction->unmapMemory(stream);
	position->unmapMemory(stream);
	tex_coord->unmapMemory(stream);

	STPShaderMemoryInternal::finish();
}