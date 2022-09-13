#include <SuperRealism+/STPExtendedScenePipeline.h>
#include <SuperTerrain+/STPCoreInfo.h>
#include <SuperRealism+/STPRealismInfo.h>

//CUDA
#include <cuda_runtime.h>
//OptiX
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

//Error
#include <SuperRealism+/Utility/STPRendererErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPMemoryError.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperRealism+/Utility/STPLogHandler.hpp>
//Engine
#include <SuperTerrain+/Utility/STPThreadPool.h>
#include <SuperTerrain+/GPGPU/STPDeviceRuntimeBinary.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
#include <SuperRealism+/Utility/STPAsyncAccelBuilder.h>
//Shader
#include <SuperRealism+/Shader/RayTracing/STPScreenSpaceRayIntersection.cuh>

#include <cassert>
//Container
#include <string>
#include <array>
#include <list>
#include <tuple>
//Thread
#include <shared_mutex>

#include <algorithm>

//GLM
#include <glm/common.hpp>

using std::array;
using std::list;
using std::pair;
using std::tuple;
using std::string;
using std::string_view;

using std::shared_mutex;
using std::unique_lock;
using std::shared_lock;
using std::future;

using std::make_pair;
using std::to_string;
using std::for_each;

using SuperTerrainPlus::STPUniqueResource;

using namespace SuperTerrainPlus::STPRealism;

//The preallocation size for log passed into OptiX compiler
constexpr static size_t mDefaultLogSize = size_t(1024);

inline static void loadExtendedShaderOption(unsigned int arch, SuperTerrainPlus::STPDeviceRuntimeBinary::STPSourceInformation& info) {
	//For performance consideration, always build with optimisation turned on;
	//we are not going to debug runtime compiled binary anyway.
	info.Option
		["-arch=sm_" + to_string(arch)]
		["-std=c++17"]
		["-rdc=true"]
		["--use_fast_math"]
		["-default-device"]
		//include directory
		["-I " + string(STPRealismInfo::OptiXInclude)]
		["-I " + string(STPRealismInfo::OptiXSDK)]
		["-I " + string(SuperTerrainPlus::STPCoreInfo::CUDAInclude)];
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

	inline void operator()(OptixModule module) const {
		STP_CHECK_OPTIX(optixModuleDestroy(module));
	}

};
using STPSmartModule = STPUniqueResource<OptixModule, nullptr, STPModuleDestroyer>;
/**
 * @brief STPProgramGroupDestroyer destroys OptiX program group.
*/
struct STPProgramGroupDestroyer {
public:

	inline void operator()(OptixProgramGroup pg) const {
		STP_CHECK_OPTIX(optixProgramGroupDestroy(pg));
	}

};
using STPSmartProgramGroup = STPUniqueResource<OptixProgramGroup, nullptr, STPProgramGroupDestroyer>;
/**
 * @brief STPPipelineDestroyer destroys OptiX pipeline.
*/
struct STPPipelineDestroyer {
public:

	inline void operator()(OptixPipeline pipeline) const {
		STP_CHECK_OPTIX(optixPipelineDestroy(pipeline));
	}

};
using STPSmartPipeline = STPUniqueResource<OptixPipeline, nullptr, STPPipelineDestroyer>;

void STPExtendedScenePipeline::STPDeviceContextDestroyer::operator()(OptixDeviceContext context) const {
	STP_CHECK_OPTIX(optixDeviceContextDestroy(context));
}

class STPExtendedScenePipeline::STPMemoryManager {
private:

	const STPExtendedScenePipeline& Master;

	//All stuff below correspond to each traceable object lives in the scene graph.
	//Such that they all have the same length and same index.
	//These pointers remain unchanged once they are allocated to avoid any headache of updating and synchronisation.
	STPSmartDeviceMemory::STPDeviceMemory<OptixInstance[]> InstanceCache;
	STPSmartDeviceMemory::STPDeviceMemory<const float* const*[]> PrimitiveGeometry;
	STPSmartDeviceMemory::STPDeviceMemory<const uint3* const*[]> PrimitiveIndex;

	//A separate stream from the renderer, so that rendering and AS build happens asynchronously.
	//Due to use of double buffering, rendering can still go through using front buffer while building in back buffer.
	STPSmartDeviceObject::STPStream ASBuildStream;
	//The renderer doesn't need to allocate a lot of memory, so keep release threshold as default zero.
	STPSmartDeviceObject::STPMemPool BuildMemoryPool;
	
	typedef list<STPExtendedSceneObject::STPTraceable::STPGeometryUpdateInformation> STPUpdateList;
	//Record all traceable objects that requested for geometry update.
	//Once updates are finished, this record will be cleared.
	STPUpdateList PendingGeometryUpdate;
	mutable shared_mutex UpdateListLock;

	STPAsyncAccelBuilder RootASBuilder;
	future<void> RootASBuildResult;

public:

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
		Master(master), ASBuildStream(STPSmartDeviceObject::makeStream(cudaStreamNonBlocking)), BuildWorker(3u) {
		cudaMemPoolProps props = { };
		props.allocType = cudaMemAllocationTypePinned;
		props.handleTypes = cudaMemHandleTypeNone;
		props.location.type = cudaMemLocationTypeDevice;
		props.location.id = 0;
		this->BuildMemoryPool = STPSmartDeviceObject::makeMemPool(props);

		//allocate memory for scene objects
		const auto [lim_traceable] = this->Master.SceneMemoryLimit;
		this->InstanceCache = STPSmartDeviceMemory::makeDevice<OptixInstance[]>(lim_traceable);
		this->PrimitiveGeometry = STPSmartDeviceMemory::makeDevice<const float* const*[]>(lim_traceable);
		this->PrimitiveIndex = STPSmartDeviceMemory::makeDevice<const uint3* const*[]>(lim_traceable);
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
		STP_CHECK_CUDA(cudaStreamSynchronize(this->ASBuildStream.get()));
	}

	/**
	 * @brief Get the traversable handle.
	 * @return The traversable handle.
	*/
	inline OptixTraversableHandle getHandle() const noexcept {
		return this->RootASBuilder.getTraversableHandle();
	}

	/**
	 * @brief Get the pointers to the primitive data array.
	 * @return The geometry vertex and index respectively, and they are all on device memory.
	*/
	inline pair<const float* const* const*, const uint3* const* const*> getPrimitiveData() const noexcept {
		return make_pair(this->PrimitiveGeometry.get(), this->PrimitiveIndex.get());
	}

	/**
	 * @brief Check if there is update available to the root level AS.
	 * If so, initiate an asynchronous build event; otherwise return.
	 * Rendering can go on as usual since build happens on a separate memory.
	*/
	void checkRootASUpdate() {
		if (this->RootASBuildResult.valid()) {
			if (this->RootASBuildResult.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
				//there is already a build operation going on
				return;
			}
			this->RootASBuildResult.get();
		}
		//no operation in progress
		{
			shared_lock read_update_lock(this->UpdateListLock);
			if (this->PendingGeometryUpdate.empty()) {
				//no update is needed
				return;
			}
		}

		//okay now start a new thread because there are pending updates
		auto tlasBuilder = [this]() -> void {
			//some commonly used data
			const cudaStream_t build_stream = this->ASBuildStream.get();
			const cudaMemPool_t build_memPool = this->BuildMemoryPool.get();
			const OptixDeviceContext context = this->Master.Context.get();
			const size_t total_instance_count = this->Master.SceneMemoryCurrent.TraceableObject;

			STPUpdateList updatingObject;
			{
				//There is a chance more pending updates are added because we released the lock after checking for emptiness.
				//That's fine, because if the list is not empty, adding more items is still non-empty.
				unique_lock splice_update_lock(this->UpdateListLock);
				//move all pending updates to our internal memory
				updatingObject.splice(updatingObject.begin(), this->PendingGeometryUpdate);
			}

			//update new instance data to the device cache
			OptixInstance* const instance_cache = this->InstanceCache.get();
			for_each(updatingObject.cbegin(), updatingObject.cend(), [instance_cache, build_stream](const auto& update_info) {
				//object ID is the same as index to the array of our objects
				const STPExtendedSceneObject::STPObjectID objID = update_info.SourceTraceable->TraceableObjectID;
				STP_CHECK_CUDA(cudaMemcpyAsync(instance_cache + objID, &update_info.Instance, sizeof(OptixInstance), cudaMemcpyHostToDevice, build_stream));
			});
			//prepare build information
			OptixAccelBuildOptions tlas_options = { };
			tlas_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			tlas_options.operation = OPTIX_BUILD_OPERATION_BUILD;
			OptixBuildInput tlas_inputs = { };
			tlas_inputs.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			tlas_inputs.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_cache);
			tlas_inputs.instanceArray.numInstances = static_cast<unsigned int>(total_instance_count);

			//allocation for temporary memory
			OptixAccelBufferSizes tlas_size = { };
			STP_CHECK_OPTIX(optixAccelComputeMemoryUsage(context, &tlas_options, &tlas_inputs, 1u, &tlas_size));
			STPSmartDeviceMemory::STPStreamedDeviceMemory<unsigned char[]> tlas_temp_mem =
				STPSmartDeviceMemory::makeStreamedDevice<unsigned char[]>(build_memPool, build_stream, tlas_size.tempSizeInBytes);

			//build TLAS
			STPAsyncAccelBuilder::STPBuildInformation tlas_build_info = { };
			tlas_build_info.Context = context;
			tlas_build_info.Stream = build_stream;
			tlas_build_info.AccelOptions = &tlas_options;
			tlas_build_info.BuildInputs = &tlas_inputs;
			tlas_build_info.numBuildInputs = 1u;
			tlas_build_info.TempBuffer = reinterpret_cast<CUdeviceptr>(tlas_temp_mem.get());
			tlas_build_info.TempBufferSize = tlas_size.tempSizeInBytes;
			tlas_build_info.OutputBufferSize = tlas_size.outputSizeInBytes;
			this->RootASBuilder.build(tlas_build_info, build_memPool);
			//now temporary buffer is done, free it after build (stream ordered)
			tlas_temp_mem.reset();

			STP_CHECK_CUDA(cudaStreamSynchronize(build_stream));
			//and now back buffer is all ready, time to swap
			//Need to ensure two things:
			// 1) There is no rendering task using this AS handle and memory, in progress and
			// 2) No rendering task should be initiated until swap buffer operation is done.
			unique_lock renderer_wait_lock(this->Master.RendererMemoryLock);
			STP_CHECK_CUDA(cudaEventSynchronize(this->Master.RendererEvent.get()));
			
			const auto prim_geometry = this->PrimitiveGeometry.get();
			const auto prim_index = this->PrimitiveIndex.get();
			//use render stream because all subsequent operations need to be visible to the renderer at the end
			const cudaStream_t render_stream = this->Master.RendererStream.get();
			for_each(updatingObject.cbegin(), updatingObject.cend(), [prim_geometry, prim_index, render_stream](const auto& update_info) {
				STPExtendedSceneObject::STPTraceable& traceable = *update_info.SourceTraceable;
				const STPExtendedSceneObject::STPObjectID objID = traceable.TraceableObjectID;

				//update geometry data
				STP_CHECK_CUDA(cudaMemcpyAsync(prim_geometry + objID, update_info.PrimitiveGeometry, sizeof(void*), cudaMemcpyHostToDevice, render_stream));
				STP_CHECK_CUDA(cudaMemcpyAsync(prim_index + objID, update_info.PrimitiveIndex, sizeof(void*), cudaMemcpyHostToDevice, render_stream));
				//swap object buffer
				traceable.swapBuffer();
			});
			this->RootASBuilder.swapHandle();
		};
		this->RootASBuildResult = this->BuildWorker.enqueue_future(tlasBuilder);
	}

};

class STPExtendedScenePipeline::STPScreenSpaceRayIntersection {
private:

	constexpr static auto SSRIShaderFilename =
		STPFile::generateFilename(STPRealismInfo::ShaderPath, "/RayTracing/STPScreenSpaceRayIntersection", ".cu");
	//shader record type
	typedef STPSbtRecord<void> STPLaunchedRayRecord;
	typedef STPSbtRecord<STPScreenSpaceRayIntersectionData::STPPrimitiveHitData> STPPrimitiveHitRecord;
	typedef STPSbtRecord<void> STPEnvironmentHitRecord;

	static_assert(STPPrimitiveHitRecord::checkDataOffset(),
		"SSRI shader binding table data has disallowed padding after the header.");

	const STPExtendedScenePipeline& Master;

	//the full pipeline containing the shader for ray intersection testing
	STPSmartPipeline IntersectionPipeline;
		//ray generation
	tuple<STPSmartDeviceMemory::STPDeviceMemory<STPLaunchedRayRecord>,
		//closest hit
		STPSmartDeviceMemory::STPDeviceMemory<STPPrimitiveHitRecord>,
		//miss
		STPSmartDeviceMemory::STPDeviceMemory<STPEnvironmentHitRecord>> IntersectionRecord;
	//the shader binding table for the intersection pipeline
	OptixShaderBindingTable IntersectionShader;

public:

	/**
	 * @brief Initialise the STPScreenSpaceRayIntersection instance.
	 * @param master The pointer to the master extended scene pipeline.
	 * @param arch Specifies the CUDA device architecture for code generation.
	*/
	STPScreenSpaceRayIntersection(const STPExtendedScenePipeline& master, unsigned int arch) : Master(master), IntersectionShader{ } {
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
		STPLogHandler::ActiveLogHandler->handle(ssri_log);
		
		//prepare for log cache
		char log[mDefaultLogSize];
		size_t logSize = mDefaultLogSize;

		OptixModule ssri_module;
		OptixPipelineCompileOptions ssri_pipeline_option = { };
		/* ------------------------------------ create module ----------------------------------- */
		{
			OptixModuleCompileOptions ssri_module_option = { };
			ssri_module_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			ssri_module_option.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;

			ssri_pipeline_option.numPayloadValues = 6;
			ssri_pipeline_option.pipelineLaunchParamsVariableName = ssri_expr.at("SSRIData").c_str();
			ssri_pipeline_option.usesPrimitiveTypeFlags =
				static_cast<decltype(OptixPipelineCompileOptions::usesPrimitiveTypeFlags)>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

			const auto [ssri_ptx, ssri_ptxSize] = STPDeviceRuntimeBinary::readPTX(ssri_program);

			STP_CHECK_OPTIX(optixModuleCreateFromPTX(this->Master.Context.get(), &ssri_module_option,
				&ssri_pipeline_option, ssri_ptx.get(), ssri_ptxSize, log, &logSize, &ssri_module));
			//We don't really care if the actual log size is larger than the default allocated size,
			//overflown log will be abandoned.
		}
		const STPSmartModule ssri_module_manager(ssri_module);
		//logging
		STPLogHandler::ActiveLogHandler->handle(string_view(log, glm::min(logSize, mDefaultLogSize)));
		//reset initial log size counter
		logSize = mDefaultLogSize;

		array<OptixProgramGroup, 3ull> ssri_program_group;
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

			STP_CHECK_OPTIX(optixProgramGroupCreate(this->Master.Context.get(), ssri_pg_desc, 3u, &ssri_pg_option,
				log, &logSize, ssri_program_group.data()));
		}
		const array<STPSmartProgramGroup, ssri_program_group.size()> ssri_pg_manager = {
			STPSmartProgramGroup(ssri_program_group[0]),
			STPSmartProgramGroup(ssri_program_group[1]),
			STPSmartProgramGroup(ssri_program_group[2])
		};
		STPLogHandler::ActiveLogHandler->handle(string_view(log, glm::min(logSize, mDefaultLogSize)));
		logSize = mDefaultLogSize;

		OptixPipeline ssri_pipeline;
		/* ----------------------------------- create pipeline --------------------------------------- */
		{
			OptixPipelineLinkOptions ssri_pipeline_link_option = { };
			ssri_pipeline_link_option.maxTraceDepth = 1u;
			ssri_pipeline_link_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

			STP_CHECK_OPTIX(optixPipelineCreate(this->Master.Context.get(), &ssri_pipeline_option, &ssri_pipeline_link_option,
				ssri_program_group.data(), static_cast<unsigned int>(ssri_program_group.size()), log, &logSize, &ssri_pipeline));
			//store the pipeline, all previously used data can be deleted, and will be done automatically
			this->IntersectionPipeline = STPSmartPipeline(ssri_pipeline);

			//stack size configuration
			OptixStackSizes ssri_stack = { };
			for (const auto& pg : ssri_program_group) {
				STP_CHECK_OPTIX(optixUtilAccumulateStackSizes(pg, &ssri_stack));
			}
			unsigned int traversal_stack, state_stack, continuation_stack;
			STP_CHECK_OPTIX(optixUtilComputeStackSizes(&ssri_stack, 1u, 0u, 0u, &traversal_stack, &state_stack, &continuation_stack));
			//our traversable structure for each scene object: GAS -> IAS with transform
			//then for each scene object, we need another IAS to merge all other IASs into one handle.
			STP_CHECK_OPTIX(optixPipelineSetStackSize(ssri_pipeline, traversal_stack, state_stack, continuation_stack, 3u));
		}
		STPLogHandler::ActiveLogHandler->handle(string_view(log, glm::min(logSize, mDefaultLogSize)));

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
	}

	STPScreenSpaceRayIntersection(const STPScreenSpaceRayIntersection&) = delete;

	STPScreenSpaceRayIntersection(STPScreenSpaceRayIntersection&&) = delete;

	STPScreenSpaceRayIntersection& operator=(const STPScreenSpaceRayIntersection&) = delete;

	STPScreenSpaceRayIntersection& operator=(STPScreenSpaceRayIntersection&&) = delete;

	~STPScreenSpaceRayIntersection() = default;

};

STPExtendedScenePipeline::~STPExtendedScenePipeline() {
	STP_CHECK_CUDA(cudaStreamSynchronize(this->RendererStream.get()));

	//kill all object dependencies
	//pending works on each object will still be running on the thread pool, and thread pool won't be dead until all tasks are finished.
	auto& [object] = this->SceneComponent;
	for_each(object.begin(), object.end(), [](auto traceable) {
		traceable->TraceableObjectID = STPExtendedSceneObject::EmptyObjectID;
		traceable->setSceneInformation(STPExtendedSceneObject::STPTraceable::STPSceneInformation { });
	});
}

void STPExtendedScenePipeline::add(STPExtendedSceneObject::STPTraceable& object) {
	size_t& curr_traceable = this->SceneMemoryCurrent.TraceableObject;
	if (curr_traceable >= this->SceneMemoryLimit.TraceableObject) {
		throw STPException::STPMemoryError("The number of traceable object has reached the limit");
	}

	//add new object an associate it with dependency of the current scene pipeline
	this->SceneComponent.TraceableObjectDatabase.emplace_back(&object);
	//the old size is the index of the newly inserted object
	object.TraceableObjectID = static_cast<STPExtendedSceneObject::STPObjectID>(curr_traceable);
	object.setSceneInformation(STPExtendedSceneObject::STPTraceable::STPSceneInformation {
		this->Context.get(),
		&this->SceneMemory->BuildWorker,
		this->SceneMemory->ObjectUpdateNotifier
	});

	curr_traceable = this->SceneComponent.TraceableObjectDatabase.size();
}