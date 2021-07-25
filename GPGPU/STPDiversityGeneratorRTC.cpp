#pragma once
#pragma warning(disable:26812)//Enum unsafe, use enum class instead
#include "STPDiversityGeneratorRTC.h"

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
using std::rethrow_exception;
using std::current_exception;
using std::exception_ptr;

using namespace SuperTerrainPlus::STPCompute;

STPDiversityGeneratorRTC::STPSourceInformation::STPSourceArgument& STPDiversityGeneratorRTC::STPSourceInformation::STPSourceArgument::addArg(const char arg[]) {
	//inserting a string literal will not cause undefined behaviour
	//string is a better choice but CUDA API only takes char array, so for simplicity store string literal.
	this->emplace_back(arg);
	return *this;
}

STPDiversityGeneratorRTC::STPLinkerInformation::STPDataJitOption& STPDiversityGeneratorRTC::STPLinkerInformation::STPDataJitOption::setDataOption(CUjit_option flag, void* value) {
	this->DataOptionFlag.emplace_back(flag);
	this->DataOptionFlagValue.emplace_back(value);
	return *this;
}

STPDiversityGeneratorRTC::STPLinkerInformation& STPDiversityGeneratorRTC::STPLinkerInformation::setLinkerOption(CUjit_option flag, void* value) {
	this->OptionFlag.emplace_back(flag);
	this->OptionFlagValue.emplace_back(value);
	return *this;
}

STPDiversityGeneratorRTC::STPLinkerInformation& STPDiversityGeneratorRTC::STPLinkerInformation::setModuleLoadOption(CUjit_option flag, void* value) {
	this->ModuleOptionFlag.emplace_back(flag);
	this->ModuleOptionFlagValue.emplace_back(value);
	return *this;
}

STPDiversityGeneratorRTC::STPLinkerInformation::STPDataJitOption& STPDiversityGeneratorRTC::STPLinkerInformation::getDataOption(string source_name) {
	//it will insert a new entry automatically if source_name is not found
	return this->DataOption[source_name];
}

STPDiversityGeneratorRTC::STPDiversityGeneratorRTC() : ModuleLoadingStatus(false), STPDiversityGenerator(){

}

STPDiversityGeneratorRTC::~STPDiversityGeneratorRTC() {
	//destroy all compiled program
	for (auto source = this->ComplicationDatabase.begin(); source != this->ComplicationDatabase.end(); source = this->ComplicationDatabase.erase(source)) {
		STPcudaCheckErr(nvrtcDestroyProgram(&source->second));
	}
	//unload module
	if (this->ModuleLoadingStatus) {
		STPcudaCheckErr(cuModuleUnload(this->GeneratorProgram));
	}
}

bool STPDiversityGeneratorRTC::attachHeader(string header_name, const string& header_code) {
	//simply add the header
	return this->ExternalHeader.emplace(header_name, header_code).second;
}

bool STPDiversityGeneratorRTC::detachHeader(string header_name) {
	return this->ExternalHeader.erase(header_name) == 1ull;
}

bool STPDiversityGeneratorRTC::attachArchive(string archive_name, string archive_filename) {
	return this->ExternalArchive.emplace(archive_name, archive_filename).second;
}

bool STPDiversityGeneratorRTC::detachArchive(string archive_name) {
	return this->ExternalArchive.erase(archive_name) == 1ull;
}

string STPDiversityGeneratorRTC::compileSource(string source_name, const string& source_code, const STPSourceInformation& source_info) {
	nvrtcProgram program;
	vector<const char*> external_header;
	vector<const char*> external_header_code;
	exception_ptr exptr;

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
	try {
		//external_header and external_header_code should have the same size
		//create the program
		STPcudaCheckErr(nvrtcCreateProgram(&program, source_code.c_str(), source_name.c_str(), static_cast<int>(external_header.size()), external_header_code.data(), external_header.data()));

		//add name expression
		for (int i = 0; i < source_info.NameExpression.size(); i++) {
			STPcudaCheckErr(nvrtcAddNameExpression(program, source_info.NameExpression[i]));
		}
		//compile program
		STPcudaCheckErr(nvrtcCompileProgram(program, static_cast<int>(source_info.Option.size()), source_info.Option.data()));
	}
	catch (...) {
		//we store the exception (if any)
		exptr = current_exception();
	}
	//and get the error message, even if exception appears
	size_t logSize;
	STPcudaCheckErr(nvrtcGetProgramLogSize(program, &logSize));
	string log;
	//reserve only increase container size but does not actually give increase the array size, so we need to fill with resize()
	log.resize(logSize);
	STPcudaCheckErr(nvrtcGetProgramLog(program, log.data()));

	if (exptr) {
		//destroy the broken program
		STPcudaCheckErr(nvrtcDestroyProgram(&program));
		throw log;
	}
	//if no error appears add the program to our database
	this->ComplicationDatabase.emplace(source_name, program);
	//return any non-error log
	return log;
}

bool STPDiversityGeneratorRTC::discardSource(string source_name) {
	return this->ComplicationDatabase.erase(source_name) == 1ull;
}

void STPDiversityGeneratorRTC::linkProgram(STPLinkerInformation& linker_info, CUjitInputType input_type) {
	CUlinkState linker;
	exception_ptr exptr;
	void* program_cubin = nullptr;

	try {
		//we can make sure the number of option flag is the same as that of value
		//create a linker
		STPcudaCheckErr(cuLinkCreate(static_cast<int>(linker_info.OptionFlag.size()), linker_info.OptionFlag.data(), linker_info.OptionFlagValue.data(), &linker));

		//for each entry, add compiled data to the linker
		for (auto compiled = this->ComplicationDatabase.cbegin(); compiled != this->ComplicationDatabase.cend(); compiled++) {
			nvrtcProgram curr_program = compiled->second;
			//get assembly code
			size_t codeSize;
			unique_ptr<char[]> code;
			switch (input_type) {
			case CU_JIT_INPUT_CUBIN:
				STPcudaCheckErr(nvrtcGetCUBINSize(curr_program, &codeSize));
				code = make_unique<char[]>(codeSize);
				STPcudaCheckErr(nvrtcGetCUBIN(curr_program, code.get()));
				break;
			case CU_JIT_INPUT_PTX:
				STPcudaCheckErr(nvrtcGetPTXSize(curr_program, &codeSize));
				code = make_unique<char[]>(codeSize);
				STPcudaCheckErr(nvrtcGetPTX(curr_program, code.get()));
				break;
			default:
				throw std::invalid_argument("unsupported input type");
				break;
			}

			//add this code to linker
			//retrieve individual linker option (if any)
			auto curr_option = linker_info.DataOption.find(compiled->first);
			if (curr_option == linker_info.DataOption.end()) {
				//no individual flag for this file
				STPcudaCheckErr(cuLinkAddData(linker, input_type, code.get(), codeSize, compiled->first.c_str(), 0, nullptr, nullptr));
			}
			else {
				//flag exists
				auto& individual_option = curr_option->second;
				STPcudaCheckErr(cuLinkAddData(linker, input_type, code.get(), codeSize, compiled->first.c_str(),
					static_cast<int>(individual_option.DataOptionFlag.size()), individual_option.DataOptionFlag.data(), individual_option.DataOptionFlagValue.data()));
			}
		}

		//for each archive, add to the linker
		for (auto archive = this->ExternalArchive.cbegin(); archive != this->ExternalArchive.cend(); archive++) {
			auto curr_option = linker_info.DataOption.find(archive->first);
			if (curr_option == linker_info.DataOption.end()) {
				STPcudaCheckErr(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, archive->second.c_str(), 0, nullptr, nullptr));
			}
			else {
				auto& archive_option = curr_option->second;
				STPcudaCheckErr(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, archive->second.c_str(),
					static_cast<int>(archive_option.DataOptionFlag.size()), archive_option.DataOptionFlag.data(), archive_option.DataOptionFlagValue.data()));
			}
		}

		//linking
		size_t cubinSize;
		STPcudaCheckErr(cuLinkComplete(linker, &program_cubin, &cubinSize));
	}
	catch (...) {
		exptr = current_exception();
		goto cleanUp;
	}
	
	//if linking is okay...
	try {
		//create module
		if (this->ModuleLoadingStatus) {
			//unload previously loaded module
			STPcudaCheckErr(cuModuleUnload(this->GeneratorProgram));
		}
		STPcudaCheckErr(cuModuleLoadDataEx(&this->GeneratorProgram, program_cubin,
			static_cast<int>(linker_info.ModuleOptionFlag.size()), linker_info.ModuleOptionFlag.data(), linker_info.ModuleOptionFlagValue.data()));
		this->ModuleLoadingStatus = true;
	}
	catch (...) {
		exptr = current_exception();
	}

	//delete the link regardlessly
	cleanUp:
	STPcudaCheckErr(cuLinkDestroy(linker));
	if (exptr) {
		//throw any module exception
		rethrow_exception(exptr);
	}
	
}

bool STPDiversityGeneratorRTC::retrieveSourceLoweredName(string source_name, STPLoweredName& expression) const {
	auto complication = this->ComplicationDatabase.find(source_name);
	if (complication == this->ComplicationDatabase.cend()) {
		//source not found
		return false;
	}
	nvrtcProgram program = complication->second;
	try {
		//get lowered name for each expression
		for (auto& expr : expression) {
			STPcudaCheckErr(nvrtcGetLoweredName(program, expr.first.c_str(), &expr.second));
		}
	}
	catch (...) {
		rethrow_exception(current_exception());
	}

	return true;
}

CUmodule STPDiversityGeneratorRTC::getGeneratorModule() const {
	return this->GeneratorProgram;
}

#pragma warning(default:26812)