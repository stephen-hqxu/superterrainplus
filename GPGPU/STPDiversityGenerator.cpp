#pragma once
#pragma warning(disable:26812)//Enum unsafe, use enum class instead
#include "STPDiversityGenerator.h"

#define STP_EXCEPTION_ON_ERROR
#include "STPDeviceErrorHandler.h"
#include <cassert>

//This is the error identifier returned by NVRTC which indicates header cannot found
#define STP_HEADER_NOT_FOUND_IDENTIFIER "catastrophic error: cannot open source file "

using std::vector;
using std::list;
using std::pair;
using std::make_pair;

using std::string;
using std::unique_ptr;
using std::make_unique;

using namespace SuperTerrainPlus::STPCompute;

STPDiversityGenerator::STPDiversityGenerator() : ModuleLoadingStatus(false) {

}

STPDiversityGenerator::~STPDiversityGenerator() {
	//destroy all compiled program
	for (auto source = this->ComplicationDatabase.begin(); source != this->ComplicationDatabase.end(); source = this->ComplicationDatabase.erase(source)) {
		STPcudaCheckErr(nvrtcDestroyProgram(&source->second));
	}
	//unload module
	if (this->ModuleLoadingStatus) {
		STPcudaCheckErr(cuModuleUnload(this->GeneratorProgram));
	}
}

bool STPDiversityGenerator::attachHeader(string header_name, const string& header_code) {
	//simply add the header
	return this->ExternalHeader.emplace(header_name, header_code).second;
}

bool STPDiversityGenerator::detachHeader(string header_name) {
	return this->ExternalHeader.erase(header_name) == 1ull;
}

unique_ptr<char[]> STPDiversityGenerator::compileSource(string source_name, const string& source_code,
	const STPStringArgument& option, const STPStringArgument& name_expressions, const STPStringArgument& extern_header) {
	nvrtcProgram program;
	vector<const char*> external_header;
	vector<const char*> external_header_code;

	//search each external header in our database
	for (auto header_name = extern_header.begin(); header_name != extern_header.end(); header_name++) {
		const auto header_code = this->ExternalHeader.find(*header_name);
		if (header_code != this->ExternalHeader.end()) {
			//code is found, add source of header
			external_header.emplace_back(*header_name);
			external_header_code.emplace_back(header_code->second.c_str());
		}
		//if not found, simply skip this header name
	}
	//external_header and external_header_code should have the same size
	//create the program
	STPcudaCheckErr(nvrtcCreateProgram(&program, source_code.c_str(), source_name.c_str(), static_cast<int>(external_header.size()), external_header_code.data(), external_header.data()));

	//add name expression
	for (int i = 0; i < name_expressions.size(); i++) {
		STPcudaCheckErr(nvrtcAddNameExpression(program, name_expressions[i]));
	}
	//compile program
	STPcudaCheckErr(nvrtcCompileProgram(program, static_cast<int>(option.size()), option.data()));

	//error message
	size_t logSize;
	STPcudaCheckErr(nvrtcGetProgramLogSize(program, &logSize));
	unique_ptr<char[]> log = make_unique<char[]>(logSize);
	STPcudaCheckErr(nvrtcGetProgramLog(program, log.get()));

	//finally add the program to our database
	this->ComplicationDatabase.emplace(source_name, program);

	return log;
}

bool STPDiversityGenerator::discardSource(string source_name) {
	return this->ComplicationDatabase.erase(source_name) == 1ull;
}

unique_ptr<char[]> STPDiversityGenerator::linkProgram(const STPJitFlag& option_flag, const STPJitFlagValue& option_value, const STPDataArgument& data_option) {
	CUlinkState linker;
	//create a linker
	//make sure the number of flag is the same as that of value
	assert(option_flag.size() == option_value.size());
	STPcudaCheckErr(cuLinkCreate(static_cast<int>(option_flag.size()), const_cast<CUjit_option*>(option_flag.data()), const_cast<void**>(option_value.data()), &linker));

	//for each entry, add compiled data to the linker
	list<unique_ptr<char[]>> source_ptx;
	for (auto compiled = this->ComplicationDatabase.cbegin(); compiled != this->ComplicationDatabase.cend(); compiled++) {
		nvrtcProgram curr_program = compiled->second;
		//get assembly code
		size_t ptxSize;
		STPcudaCheckErr(nvrtcGetPTXSize(curr_program, &ptxSize));
		unique_ptr<char[]> ptx = make_unique<char[]>(ptxSize);
		STPcudaCheckErr(nvrtcGetPTX(curr_program, ptx.get()));
		//add this code to linker
		//retrieve individual linker option (if any)
		const auto& curr_option = data_option.find(compiled->first);
		if (curr_option == data_option.end()) {
			//no individual flag for this file
			STPcudaCheckErr(cuLinkAddData(linker, CU_JIT_INPUT_PTX, ptx.get(), ptxSize, compiled->first.c_str(), 0, nullptr, nullptr));
		}
		else {
			//flag exists
			const auto& individual_option = curr_option->second;
			assert(individual_option.first.size() == individual_option.second.size());
			STPcudaCheckErr(cuLinkAddData(linker, CU_JIT_INPUT_PTX, ptx.get(), ptxSize, compiled->first.c_str(), 
				static_cast<int>(individual_option.first.size()), const_cast<CUjit_option*>(individual_option.first.data()), const_cast<void**>(individual_option.second.data())));
		}

		//we need to retain the data until linking is completed
		source_ptx.emplace_back(std::move(ptx));
	}

	//linking
	size_t cubinSize;
	void* program_cubin;
	STPcudaCheckErr(cuLinkComplete(linker, &program_cubin, &cubinSize));
	//create module
	STPcudaCheckErr(cuModuleLoadData(&this->GeneratorProgram, program_cubin));
	this->ModuleLoadingStatus = true;

	//finally
	STPcudaCheckErr(cuLinkDestroy(linker));

	return nullptr;
}

#pragma warning(default:26812)