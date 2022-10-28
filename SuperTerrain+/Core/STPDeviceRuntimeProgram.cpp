#include <SuperTerrain+/GPGPU/STPDeviceRuntimeProgram.h>

//Error
#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

#include <algorithm>

using namespace SuperTerrainPlus;

/**
 * @brief Delete a link state.
*/
struct STPLinkStateDeleter {
public:

	inline void operator()(CUlinkState link) const {
		STP_CHECK_CUDA(cuLinkDestroy(link));
	}

};

STPDeviceRuntimeProgram::STPLinkerInformation::STPDataJitOption&
	STPDeviceRuntimeProgram::STPLinkerInformation::STPDataJitOption::operator()(CUjit_option flag, void* value) {
	this->OptionFlag.emplace_back(flag);
	this->OptionValue.emplace_back(value);
	return *this;
}

void STPDeviceRuntimeProgram::STPModuleDeleter::operator()(CUmodule module) const {
	STP_CHECK_CUDA(cuModuleUnload(module));
}

STPDeviceRuntimeProgram::STPSmartModule STPDeviceRuntimeProgram::link(STPLinkerInformation& linker_info) {
	auto& [linker_opt, data_opt, archive_opt, module_opt] = linker_info;

	//create a linker
	CUlinkState linker;
	STP_CHECK_CUDA(cuLinkCreate(static_cast<int>(linker_opt.OptionFlag.size()), linker_opt.OptionFlag.data(),
		linker_opt.OptionValue.data(), &linker));
	const STPUniqueResource<CUlinkState, nullptr, STPLinkStateDeleter> managed_linker(linker);

	//add each data to the linker
	for (auto& [binary, bin_type, data_jit] : data_opt) {
		const nvrtcProgram program = binary->Program.get();
		//get code based on user specified binary type
		STPDeviceRuntimeBinary::STPProgramData program_data;
		CUjitInputType bin_input = { };
		switch (bin_type) {
		case STPBinaryType::PTX:
			program_data = STPDeviceRuntimeBinary::readPTX(program);
			bin_input = CU_JIT_INPUT_PTX;
			break;
		}

		//add to the linker
		//we can safely delete the code after it has been added
		const auto& [code, codeSize] = program_data;
		STP_CHECK_CUDA(cuLinkAddData(linker, bin_input, code.get(), codeSize, binary->Identifier.c_str(),
			static_cast<unsigned int>(data_jit.OptionFlag.size()), data_jit.OptionFlag.data(), data_jit.OptionValue.data()));
	}

	//add archive
	std::for_each(archive_opt.begin(), archive_opt.end(), [linker](auto& arc) {
		auto& [filename, opt] = arc;

		STP_CHECK_CUDA(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, filename.c_str(),
			static_cast<unsigned int>(opt.OptionFlag.size()), opt.OptionFlag.data(), opt.OptionValue.data()));
	});

	//link
	size_t cubinSize;
	void* program_cubin;
	STP_CHECK_CUDA(cuLinkComplete(linker, &program_cubin, &cubinSize));

	//create module
	CUmodule module;
	STP_CHECK_CUDA(cuModuleLoadDataEx(&module, program_cubin,
		static_cast<unsigned int>(module_opt.OptionFlag.size()), module_opt.OptionFlag.data(), module_opt.OptionValue.data()));
	
	return STPSmartModule(module);
}