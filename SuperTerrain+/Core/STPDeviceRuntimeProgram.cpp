#include <SuperTerrain+/GPGPU/STPDeviceRuntimeProgram.h>

//Error
#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

#include <algorithm>

using std::unique_ptr;
using std::make_unique;

using namespace SuperTerrainPlus;

/**
 * @brief Delete a link state.
*/
struct STPLinkStateDeleter {
public:

	inline void operator()(CUlinkState link) {
		STPcudaCheckErr(cuLinkDestroy(link));
	}

};

STPDeviceRuntimeProgram::STPLinkerInformation::STPDataJitOption&
	STPDeviceRuntimeProgram::STPLinkerInformation::STPDataJitOption::operator()(CUjit_option flag, void* value) {
	this->OptionFlag.emplace_back(flag);
	this->OptionValue.emplace_back(value);
	return *this;
}

void STPDeviceRuntimeProgram::STPModuleDeleter::operator()(CUmodule module) const {
	STPcudaCheckErr(cuModuleUnload(module));
}

CUmodule STPDeviceRuntimeProgram::operator*() const {
	return this->Module.get();
}

void STPDeviceRuntimeProgram::linkFromBinary(STPLinkerInformation& linker_info) {
	auto& [linker_opt, data_opt, archive_opt, module_opt] = linker_info;

	//create a linker
	CUlinkState linker;
	STPcudaCheckErr(cuLinkCreate(static_cast<int>(linker_opt.OptionFlag.size()), linker_opt.OptionFlag.data(),
		linker_opt.OptionValue.data(), &linker));
	unique_ptr<std::remove_pointer_t<CUlinkState>, STPLinkStateDeleter> managed_linker(linker);

	//add each data to the linker
	for (auto& [binary, bin_type, data_jit] : data_opt) {
		const nvrtcProgram program = **binary;
		//get code based on user specified binary type
		size_t codeSize = 0ull;
		unique_ptr<char[]> code;
		CUjitInputType bin_input = { };
		switch (bin_type) {
		case STPBinaryType::PTX:
			STPcudaCheckErr(nvrtcGetPTXSize(program, &codeSize));
			code = make_unique<char[]>(codeSize);
			STPcudaCheckErr(nvrtcGetPTX(program, code.get()));

			bin_input = CU_JIT_INPUT_PTX;
			break;
		case STPBinaryType::CUBIN:
			STPcudaCheckErr(nvrtcGetCUBINSize(program, &codeSize));
			code = make_unique<char[]>(codeSize);
			STPcudaCheckErr(nvrtcGetCUBIN(program, code.get()));

			bin_input = CU_JIT_INPUT_CUBIN;
			break;
		}

		//add to the linker
		//we can safely delete the code after it has been added
		STPcudaCheckErr(cuLinkAddData(linker, bin_input, code.get(), codeSize, binary->name().c_str(),
			static_cast<unsigned int>(data_jit.OptionFlag.size()), data_jit.OptionFlag.data(), data_jit.OptionValue.data()));
	}

	//add archive
	std::for_each(archive_opt.begin(), archive_opt.end(), [linker](auto& arc) {
		auto& [filename, opt] = arc;

		STPcudaCheckErr(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, filename.c_str(),
			static_cast<unsigned int>(opt.OptionFlag.size()), opt.OptionFlag.data(), opt.OptionValue.data()));
	});

	//link
	size_t cubinSize;
	void* program_cubin;
	STPcudaCheckErr(cuLinkComplete(linker, &program_cubin, &cubinSize));

	//create module
	CUmodule module;
	STPcudaCheckErr(cuModuleLoadDataEx(&module, program_cubin,
		static_cast<unsigned int>(module_opt.OptionFlag.size()), module_opt.OptionFlag.data(), module_opt.OptionValue.data()));
	//store
	this->Module = std::move(STPManagedModule(module));
}