#include <SuperRealism+/STPExtendedScenePipeline.h>
#include <SuperTerrain+/STPCoreInfo.h>
#include <SuperRealism+/STPRealismInfo.h>

//CUDA
#include <cuda_runtime.h>
//OptiX
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

#pragma warning(push)
#pragma warning(disable: 4996)//use strcat_s instead of strcat for security reason, which is not my fault but OptiX's...
//Error
#include <SuperRealism+/Utility/STPRendererErrorHandler.hpp>
#pragma warning(pop)
//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperRealism+/Utility/STPLogHandler.hpp>
//Engine
#include <SuperTerrain+/GPGPU/STPDeviceRuntimeBinary.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//Shader
#include <SuperRealism+/Shader/RayTracing/STPScreenSpaceRayIntersection.cuh>

//System
#include <string>
#include <array>
#include <tuple>

using std::array;
using std::tuple;
using std::string;
using std::to_string;
using std::make_unique;
using std::remove_pointer_t;
using std::unique_ptr;

using namespace SuperTerrainPlus::STPRealism;

//The preallocation size for log passed into OptiX compiler
constexpr static size_t mDefaultLogSize = 1024ull;

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

/**
 * @brief The base shader binding record for packing.
 * @tparam T The data section of the shader record.
*/
template<class T>
struct STPSbtRecord {
public:

	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char Header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T Data;

};

/**
 * @brief STPModuleDestroyer destroys OptiX module.
*/
struct STPModuleDestroyer {
public:

	inline void operator()(OptixModule module) const {
		STP_CHECK_OPTIX(optixModuleDestroy(module));
	}

};
using STPSmartModule = unique_ptr<remove_pointer_t<OptixModule>, STPModuleDestroyer>;
/**
 * @brief STPProgramGroupDestroyer destroys OptiX program group.
*/
struct STPProgramGroupDestroyer {
public:

	inline void operator()(OptixProgramGroup pg) const {
		STP_CHECK_OPTIX(optixProgramGroupDestroy(pg));
	}

};
using STPSmartProgramGroup = unique_ptr<remove_pointer_t<OptixProgramGroup>, STPProgramGroupDestroyer>;
/**
 * @brief STPPipelineDestroyer destroys OptiX pipeline.
*/
struct STPPipelineDestroyer {
public:

	inline void operator()(OptixPipeline pipeline) const {
		STP_CHECK_OPTIX(optixPipelineDestroy(pipeline));
	}

};
using STPSmartPipeline = unique_ptr<remove_pointer_t<OptixPipeline>, STPPipelineDestroyer>;

void STPExtendedScenePipeline::STPDeviceContextDestroyer::operator()(OptixDeviceContext context) const {
	STP_CHECK_OPTIX(optixDeviceContextDestroy(context));
}

class STPExtendedScenePipeline::STPScreenSpaceRayIntersection {
private:

	constexpr static auto SSRIShaderFilename =
		STPFile::generateFilename(STPRealismInfo::ShaderPath, "/RayTracing/STPScreenSpaceRayIntersection", ".cu");
	//shader record type
	typedef STPSbtRecord<STPScreenSpaceRayIntersectionData::STPLaunchedRayData> STPLaunchedRayRecord;
	typedef STPSbtRecord<STPScreenSpaceRayIntersectionData::STPPrimitiveHitData> STPPrimitiveHitRecord;
	typedef STPSbtRecord<STPScreenSpaceRayIntersectionData::STPEnvironmentHitData> STPEnvironmentHitRecord;

	STPExtendedScenePipeline& Master;

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
	 * @param master The pointer to the master extended scene pipeline
	 * @param arch Specifies the CUDA device architecture for code generation.
	*/
	STPScreenSpaceRayIntersection(STPExtendedScenePipeline& master, unsigned int arch) : Master(master), IntersectionShader{ } {
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
		string log;
		size_t logSize = mDefaultLogSize;
		log.resize(logSize);

		OptixModule ssri_module;
		OptixPipelineCompileOptions ssri_pipeline_option = { };
		/* ------------------------------------ create module ----------------------------------- */
		{
			OptixModuleCompileOptions ssri_module_option = { };
			ssri_module_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			ssri_module_option.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;

			ssri_pipeline_option.numPayloadValues = 6;
			ssri_pipeline_option.pipelineLaunchParamsVariableName = ssri_expr.at("SSRIData").c_str();
			ssri_pipeline_option.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

			const auto [ssri_ptx, ssri_ptxSize] = STPDeviceRuntimeBinary::readPTX(ssri_program);

			STP_CHECK_OPTIX(optixModuleCreateFromPTX(this->Master.Context.get(), &ssri_module_option,
				&ssri_pipeline_option, ssri_ptx.get(), ssri_ptxSize, log.data(), &logSize, &ssri_module));
			//We don't really care if the actual log size is larger than the default allocated size,
			//overflown log will be abandoned.
		}
		const STPSmartModule ssri_module_manager(ssri_module);
		//logging
		STPLogHandler::ActiveLogHandler->handle(log);

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
				log.data(), &logSize, ssri_program_group.data()));
		}
		const array<STPSmartProgramGroup, 3ull> ssri_pg_manager = {
			STPSmartProgramGroup(ssri_program_group[0]),
			STPSmartProgramGroup(ssri_program_group[1]),
			STPSmartProgramGroup(ssri_program_group[2])
		};
		STPLogHandler::ActiveLogHandler->handle(log);

		OptixPipeline ssri_pipeline;
		/* ----------------------------------- create pipeline --------------------------------------- */
		{
			OptixPipelineLinkOptions ssri_pipeline_link_option = {};
			ssri_pipeline_link_option.maxTraceDepth = 1u;
			ssri_pipeline_link_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

			STP_CHECK_OPTIX(optixPipelineCreate(this->Master.Context.get(), &ssri_pipeline_option,
				&ssri_pipeline_link_option, ssri_program_group.data(), 3u, log.data(), &logSize, &ssri_pipeline));
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
		STPLogHandler::ActiveLogHandler->handle(log);

		/* ---------------------------- allocate shader binding table -------------------------------- */
		{
			auto& [ssri_rec_rg, ssri_rec_ch, ssri_rec_ms] = this->IntersectionRecord;
			ssri_rec_rg = STPSmartDeviceMemory::makeDevice<STPLaunchedRayRecord>();
			ssri_rec_ch = STPSmartDeviceMemory::makeDevice<STPPrimitiveHitRecord>();
			ssri_rec_ms = STPSmartDeviceMemory::makeDevice<STPEnvironmentHitRecord>();
			//header packing
			STPLaunchedRayRecord rec_rg = { };
			STPPrimitiveHitRecord rec_ch = { };
			STPEnvironmentHitRecord rec_ms = { };
			STP_CHECK_OPTIX(optixSbtRecordPackHeader(ssri_program_group[0], &rec_rg));
			STP_CHECK_OPTIX(optixSbtRecordPackHeader(ssri_program_group[1], &rec_ch));
			STP_CHECK_OPTIX(optixSbtRecordPackHeader(ssri_program_group[2], &rec_ms));
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

STPExtendedScenePipeline::~STPExtendedScenePipeline() = default;