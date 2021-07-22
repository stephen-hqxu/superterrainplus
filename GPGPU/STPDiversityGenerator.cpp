#pragma once
#pragma warning(disable:26812)//Enum unsafe, use enum class instead
#include "STPDiversityGenerator.h"

#define STP_EXCEPTION_ON_ERROR
#include "STPDeviceErrorHandler.h"
#include <stdexcept>

using std::vector;
using std::list;
using std::pair;
using std::make_pair;

using std::string;
using std::unique_ptr;
using std::make_unique;

using namespace SuperTerrainPlus::STPCompute;

STPDiversityGenerator::STPSourceInformation::STPSourceArgument& STPDiversityGenerator::STPSourceInformation::STPSourceArgument::addArg(const char arg[]) {
	//inserting a string literal will not cause undefined behaviour
	//string is a better choice but CUDA API only takes char array, so for simplicity store string literal.
	this->emplace_back(arg);
	return *this;
}

STPDiversityGenerator::STPLinkerInformation::STPDataJitOption& STPDiversityGenerator::STPLinkerInformation::STPDataJitOption::setDataOption(CUjit_option flag, void* value) {
	this->DataOptionFlag.emplace_back(flag);
	this->DataOptionFlagValue.emplace_back(value);
	return *this;
}

STPDiversityGenerator::STPLinkerInformation& STPDiversityGenerator::STPLinkerInformation::setLinkerOption(CUjit_option flag, void* value) {
	this->OptionFlag.emplace_back(flag);
	this->OptionFlagValue.emplace_back(value);
	return *this;
}

STPDiversityGenerator::STPLinkerInformation& STPDiversityGenerator::STPLinkerInformation::setModuleLoadOption(CUjit_option flag, void* value) {
	this->ModuleOptionFlag.emplace_back(flag);
	this->ModuleOptionFlagValue.emplace_back(value);
	return *this;
}

STPDiversityGenerator::STPLinkerInformation::STPDataJitOption& STPDiversityGenerator::STPLinkerInformation::getDataOption(string source_name) {
	//it will insert a new entry automatically if source_name is not found
	return this->DataOption[source_name];
}

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

STPDiversityGenerator::STPGeneratorLog STPDiversityGenerator::compileSource(string source_name, const string& source_code, const STPSourceInformation& source_info) {
	nvrtcProgram program;
	vector<const char*> external_header;
	vector<const char*> external_header_code;

	//search each external header in our database
	for (auto header_name = source_info.ExternalHeader.begin(); header_name != source_info.ExternalHeader.end(); header_name++) {
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
	for (int i = 0; i < source_info.NameExpression.size(); i++) {
		STPcudaCheckErr(nvrtcAddNameExpression(program, source_info.NameExpression[i]));
	}
	//compile program
	STPcudaCheckErr(nvrtcCompileProgram(program, static_cast<int>(source_info.Option.size()), source_info.Option.data()));

	//error message
	size_t logSize;
	STPcudaCheckErr(nvrtcGetProgramLogSize(program, &logSize));
	STPGeneratorLog log = make_unique<char[]>(logSize);
	STPcudaCheckErr(nvrtcGetProgramLog(program, log.get()));

	//finally add the program to our database
	this->ComplicationDatabase.emplace(source_name, program);

	return log;
}

bool STPDiversityGenerator::discardSource(string source_name) {
	return this->ComplicationDatabase.erase(source_name) == 1ull;
}

void STPDiversityGenerator::linkProgram(STPLinkerInformation& linker_info) {
	CUlinkState linker;
	//we can make sure the number of option flag is the same as that of value
	//create a linker
	STPcudaCheckErr(cuLinkCreate(static_cast<int>(linker_info.OptionFlag.size()), linker_info.OptionFlag.data(), linker_info.OptionFlagValue.data(), &linker));

	//for each entry, add compiled data to the linker
	for (auto compiled = this->ComplicationDatabase.cbegin(); compiled != this->ComplicationDatabase.cend(); compiled++) {
		nvrtcProgram curr_program = compiled->second;
		//get assembly code
		size_t ptxSize;
		STPcudaCheckErr(nvrtcGetPTXSize(curr_program, &ptxSize));
		unique_ptr<char[]> ptx = make_unique<char[]>(ptxSize);
		STPcudaCheckErr(nvrtcGetPTX(curr_program, ptx.get()));
		//add this code to linker
		
		//retrieve individual linker option (if any)
		auto curr_option = linker_info.DataOption.find(compiled->first);
		if (curr_option == linker_info.DataOption.end()) {
			//no individual flag for this file
			STPcudaCheckErr(cuLinkAddData(linker, CU_JIT_INPUT_PTX, ptx.get(), ptxSize, compiled->first.c_str(), 0, nullptr, nullptr));
		}
		else {
			//flag exists
			auto& individual_option = curr_option->second;
			STPcudaCheckErr(cuLinkAddData(linker, CU_JIT_INPUT_PTX, ptx.get(), ptxSize, compiled->first.c_str(), 
				static_cast<int>(individual_option.DataOptionFlag.size()), individual_option.DataOptionFlag.data(), individual_option.DataOptionFlagValue.data()));
		}
	}

	//linking
	size_t cubinSize;
	void* program_cubin;
	STPcudaCheckErr(cuLinkComplete(linker, &program_cubin, &cubinSize));
	//create module
	if (this->ModuleLoadingStatus) {
		//unload previously loaded module
		STPcudaCheckErr(cuModuleUnload(this->GeneratorProgram));
	}
	STPcudaCheckErr(cuModuleLoadDataEx(&this->GeneratorProgram, program_cubin, 
		static_cast<int>(linker_info.ModuleOptionFlag.size()), linker_info.ModuleOptionFlag.data(), linker_info.ModuleOptionFlagValue.data()));
	this->ModuleLoadingStatus = true;

	//finally
	STPcudaCheckErr(cuLinkDestroy(linker));
}

bool STPDiversityGenerator::retrieveSourceLoweredName(string source_name, STPLoweredName& expression) const {
	auto complication = this->ComplicationDatabase.find(source_name);
	if (complication == this->ComplicationDatabase.cend()) {
		//source not found
		return false;
	}
	nvrtcProgram program = complication->second;
	//get lowered name for each expression
	for (auto& expr : expression) {
		STPcudaCheckErr(nvrtcGetLoweredName(program, expr.first.c_str(), &expr.second));
	}

	return true;
}

CUmodule STPDiversityGenerator::getGeneratorModule() const {
	return this->GeneratorProgram;
}

#pragma warning(default:26812)