#pragma once
#ifndef _STP_AMBIENT_OCCLUSION_H_
#define _STP_AMBIENT_OCCLUSION_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Screen
#include "STPScreen.h"
//GL Object
#include "../../Object/STPSampler.h"
#include "../../Object/STPTexture.h"
#include "../../Object/STPBindlessTexture.h"

#include "../../Environment/STPOcclusionKernelSetting.h"
//Output Processing
#include "STPGaussianFilter.h"

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAmbientOcclusion is an indirect lighting approximation that 
	 * tries to approximate indirect lighting by darkening creases, holes and surfaces that are closed to each other.
	*/
	class STP_REALISM_API STPAmbientOcclusion : private STPScreen {
	public:

		/**
		 * @brief STPOcclusionAlgorithm provides selections to available ambient occlusion algorithms.
		*/
		enum class STPOcclusionAlgorithm : unsigned char {
			//The legacy ambient occlusion. 
			//It performs Monte-Carol sampling around the pixel and counts how many samples within a hemisphere falls behind the geometries
			//to simulate occlusion of indirect lights.
			SSAO = 0x00u,
			//An improved ambient occlusion over the legacy algorithm.
			//This algorithm scans through the space above the geometry horizon and perform ray marching for each ray.
			//Then determines the distance to the closest hit as an occlusion factor.
			//Compare to legacy AO the occlusion factor is continuous.
			HBAO = 0x01u,
		};

		/**
		 * @brief STPOcclusionKernel defines how ambient occlusion should behave based the occlusion algorithm chosen.
		*/
		template<STPOcclusionAlgorithm O>
		struct STPOcclusionKernel;

		/**
		 * @brief STPOcclusionKernelInstance specifies a base instance of all ambient occlusion kernel instances.
		*/
		class STP_REALISM_API STPOcclusionKernelInstance {
		private:

			friend class STPAmbientOcclusion;

			typedef STPShaderManager::STPShaderSource::STPMacroValueDictionary STPKernelOption;
			//The RNG should return a uniformly distributed random real number between 0 and 1
			typedef std::function<float(void)> STPKernelRNG;

			/**
			 * @brief Generate the random rotation vector texture.
			 * @param texture The texture where rotation vector data should be stored to.
			 * @param rng The random number generator which returns a float.
			*/
			virtual void rotationVector(STPTexture&, const STPKernelRNG&) const = 0;

			/**
			 * @brief Load shader compiler options for the current kernel instance.
			 * @param option The pointer to shader compiler option to be loaded.
			*/
			virtual void compilerOption(STPKernelOption&) const = 0;

			/**
			 * @brief Send kernel information as uniforms to the shader.
			 * @param program The program where data should be sent to.
			 * @param rng The random number generator which returns a float.
			*/
			virtual void uniformKernel(STPProgramManager&, const STPKernelRNG&) const = 0;

		public:

			//The type of kernel instance.
			const STPOcclusionAlgorithm Occluder;
			//The pointer to the base setting for the occlusion kernel.
			const STPEnvironment::STPOcclusionKernelSetting& Kernel;

			/**
			 * @brief Init a new kernel instance.
			 * @param kernel_setting The pointer to the setting to configure how ambient occlusion will be performed.
			 * The pointer will be retained until the instance is destroyed.
			 * @param algorithm The occlusion algorithm for this kernel instance.
			*/
			STPOcclusionKernelInstance(const STPEnvironment::STPOcclusionKernelSetting&, STPOcclusionAlgorithm);

			STPOcclusionKernelInstance(const STPOcclusionKernelInstance&) = default;

			STPOcclusionKernelInstance(STPOcclusionKernelInstance&&) noexcept = default;

			STPOcclusionKernelInstance& operator=(const STPOcclusionKernelInstance&) = delete;

			STPOcclusionKernelInstance& operator=(STPOcclusionKernelInstance&&) = delete;

			virtual ~STPOcclusionKernelInstance() = default;

		};

	private:

		STPTexture RandomRotationVector;
		STPBindlessTexture RandomRotationVectorHandle;
		//Sampler for the external input
		STPSampler GBufferSampler;

		//ambient occlusion output
		mutable STPSimpleScreenFrameBuffer OcclusionResultContainer;

		//Store the dimension of the texture which contains random rotation vectors
		const glm::uvec2 NoiseDimension;

		STPGaussianFilter BlurWorker;

	public:

		/**
		 * @brief Initialise a new ambient occlusion rendering component.
		 * @param kernel_instance Specifies a kernel instance which selects the ambinet occlusion algorithm to be used.
		 * The instance will not be retained by the object, it can be safely destroyed by the caller afterwards.
		 * @param filter A rvalue reference to a Gaussian filter which will be used to blurred the output ambient occlusion.
		 * @param kernel_init The pointer to ambient occlusion initialiser.
		*/
		STPAmbientOcclusion(const STPOcclusionKernelInstance&, STPGaussianFilter&&, const STPScreenInitialiser&);

		STPAmbientOcclusion(const STPAmbientOcclusion&) = delete;

		STPAmbientOcclusion(STPAmbientOcclusion&&) = delete;

		STPAmbientOcclusion& operator=(const STPAmbientOcclusion&) = delete;

		STPAmbientOcclusion& operator=(STPAmbientOcclusion&&) = delete;

		~STPAmbientOcclusion() = default;

		/**
		 * @brief Set new screen space dimension and re-initialise ambient occlusion framebuffer.
		 * @param stencil The pointer to the stencil buffer to be used to control ambient occlusion region, or nullptr if not used.
		 * @param dimension The new dimension for the rendering screen.
		*/
		void setScreenSpace(STPTexture*, glm::uvec2);

		/**
		 * @brief Start performing ambient occlusion calculation.
		 * The occlusion result is stored in the internal buffer, this call clears all previous data.
		 * @param depth The texture which contains geometry depth data.
		 * This depth data will be used to reconstruct geometry position.
		 * @param normal The texture which contains geometry normal data.
		 * @param output The pointer to the framebuffer where the final ambient occlusion output will be stored.
		*/
		void occlude(const STPTexture&, const STPTexture&, STPFrameBuffer&) const;

	};

#define AO_KERNEL_DEF(ALG) \
template<> struct STP_REALISM_API STPAmbientOcclusion::STPOcclusionKernel<STPAmbientOcclusion::STPOcclusionAlgorithm::ALG> : public STPAmbientOcclusion::STPOcclusionKernelInstance

	AO_KERNEL_DEF(SSAO) {
	private:

		void rotationVector(STPTexture&, const STPKernelRNG&) const override;

		void compilerOption(STPKernelOption&) const override;

		void uniformKernel(STPProgramManager&, const STPKernelRNG&) const override;

	public:

		STPOcclusionKernel(const STPEnvironment::STPOcclusionKernelSetting&);

		~STPOcclusionKernel() = default;

		//Determine how many samples to be taken each pixels,
		//more samples give less noise but it is also more computationally expensive.
		unsigned int KernelSize;

	};

	AO_KERNEL_DEF(HBAO) {
	private:

		void rotationVector(STPTexture&, const STPKernelRNG&) const override;

		//no compiler option for HBAO
		void compilerOption(STPKernelOption&) const override { };

		void uniformKernel(STPProgramManager&, const STPKernelRNG&) const override;

	public:

		STPOcclusionKernel(const STPEnvironment::STPOcclusionKernelSetting&);

		~STPOcclusionKernel() = default;

		//Specifies the number of ray above the horizon to scan through,
		//and the number of segment on the marched ray, respectively.
		//Greater number of steps give better approximation with exchange of slower runtime performance.
		unsigned int DirectionStep, RayStep;

	};

#undef AO_KERNEL_DEF

}
#endif//_STP_AMBIENT_OCCLUSION_H_