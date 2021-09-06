#pragma once
#pragma warning(disable:26812)//Enum unsafe, use enum class instead
#include <SuperTerrain+/GPGPU/STPDiversityGeneratorRTC.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPSerialisationError.h>
#include <SuperTerrain+/Utility/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

//IO
#include <fstream>
#include <sstream>

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

STPDiversityGeneratorRTC::STPSourceInformation::STPSourceArgument& STPDiversityGeneratorRTC::STPSourceInformation::STPSourceArgument::operator[](const char arg[]) {
	//inserting a string literal will not cause undefined behaviour
	//string is a better choice but CUDA API only takes char array, so for simplicity store string literal.
	this->emplace_back(arg);
	return *this;
}

STPDiversityGeneratorRTC::STPLinkerInformation::STPDataJitOption& STPDiversityGeneratorRTC::STPLinkerInformation::STPDataJitOption::operator()(CUjit_option flag, void* value) {
	this->OptionFlag.emplace_back(flag);
	this->OptionValue.emplace_back(value);
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
	for (auto source = this->CompilationDatabase.begin(); source != this->CompilationDatabase.end(); source = this->CompilationDatabase.erase(source)) {
		STPcudaCheckErr(nvrtcDestroyProgram(&source->second));
	}
	//unload module
	if (this->ModuleLoadingStatus) {
		STPcudaCheckErr(cuModuleUnload(this->GeneratorProgram));
	}
}

string STPDiversityGeneratorRTC::readSource(string filename) {
	using std::ifstream;
	using std::stringstream;

	ifstream source_reader(filename);
	if (!source_reader) {
		throw STPException::STPSerialisationError(("file '" + filename + "' cannot be opened.").c_str());
	}
	//read all lines
	stringstream buffer;
	buffer << source_reader.rdbuf();

	return buffer.str();
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
	//make sure the source name is unique
	if (this->CompilationDatabase.find(source_name) != this->CompilationDatabase.end()) {
		throw STPException::STPMemoryError(
			(string(__FILE__) + "::" + string(__FUNCTION__) + "\nA duplicate source with name '" + source_name + "' has been compiled before.").c_str()
		);
	}

	nvrtcProgram program;
	vector<const char*> external_header;
	vector<const char*> external_header_code;
	exception_ptr exptr;

	//search each external header in our database
	for (auto header_name = source_info.ExternalHeader.begin(); header_name != source_info.ExternalHeader.end(); header_name++) {
		if (const auto header_code = this->ExternalHeader.find(*header_name); 
			header_code != this->ExternalHeader.end()) {
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
		for (auto expr : source_info.NameExpression) {
			STPcudaCheckErr(nvrtcAddNameExpression(program, expr));
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
		throw STPException::STPCompilationError(log.c_str());
	}
	//if no error appears grab the lowered name expression from compiled program
	STPLoweredName& current_source_name = this->CompilationNameDatabase[source_name];
	for (auto expr : source_info.NameExpression) {
		if (const char* current_lowered_name; 
			nvrtcGetLoweredName(program, expr, &current_lowered_name) == NVRTC_SUCCESS) {
			//we got the name, add to the database
			current_source_name[expr] = current_lowered_name;
		}
		//lowered name cannot be found in the source code
		//simply ignore this name
	}
	//and finally add the program to our database
	this->CompilationDatabase.emplace(source_name, program);
	//return any non-error log
	return log;
}

bool STPDiversityGeneratorRTC::discardSource(string source_name) {
	if (auto source = this->CompilationDatabase.find(source_name); 
		source != this->CompilationDatabase.end()) {
		//make sure the source exists
		STPcudaCheckErr(nvrtcDestroyProgram(&source->second));
		//and destroy the program effectively
		this->CompilationDatabase.erase(source);
		this->CompilationNameDatabase.erase(source_name);
		return true;
	}
	return false;
}

void STPDiversityGeneratorRTC::linkProgram(STPLinkerInformation& linker_info, CUjitInputType input_type) {
	CUlinkState linker;
	exception_ptr exptr;
	void* program_cubin = nullptr;

	try {
		//we can make sure the number of option flag is the same as that of value
		//create a linker
		STPLinkerInformation::STPDataJitOption& linkerOption = linker_info.LinkerOption;
		STPcudaCheckErr(cuLinkCreate(static_cast<int>(linkerOption.OptionFlag.size()), linkerOption.OptionFlag.data(), linkerOption.OptionValue.data(), &linker));

		//for each entry, add compiled data to the linker
		for (auto compiled = this->CompilationDatabase.cbegin(); compiled != this->CompilationDatabase.cend(); compiled++) {
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
				throw STPException::STPUnsupportedFunctionality("unsupported assembly code type");
				break;
			}

			//add this code to linker
			//retrieve individual linker option (if any)
			//we can safely delete the code since:
			//'Ownership of data is retained by the caller. No reference is retained to any inputs after this call returns.'
			if (auto curr_option = linker_info.DataOption.find(compiled->first); 
				curr_option == linker_info.DataOption.end()) {
				//no individual flag for this file
				STPcudaCheckErr(cuLinkAddData(linker, input_type, code.get(), codeSize, compiled->first.c_str(), 0, nullptr, nullptr));
			}
			else {
				//flag exists
				auto& individual_option = curr_option->second;
				STPcudaCheckErr(cuLinkAddData(linker, input_type, code.get(), codeSize, compiled->first.c_str(),
					static_cast<int>(individual_option.OptionFlag.size()), individual_option.OptionFlag.data(), individual_option.OptionValue.data()));
			}
		}

		//for each archive, add to the linker
		for (auto archive = this->ExternalArchive.cbegin(); archive != this->ExternalArchive.cend(); archive++) {
			if (auto curr_option = linker_info.DataOption.find(archive->first); 
				curr_option == linker_info.DataOption.end()) {
				STPcudaCheckErr(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, archive->second.c_str(), 0, nullptr, nullptr));
			}
			else {
				auto& archive_option = curr_option->second;
				STPcudaCheckErr(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, archive->second.c_str(),
					static_cast<int>(archive_option.OptionFlag.size()), archive_option.OptionFlag.data(), archive_option.OptionValue.data()));
			}
		}

		//linking
		size_t cubinSize;
		STPcudaCheckErr(cuLinkComplete(linker, &program_cubin, &cubinSize));

		//create module
		if (this->ModuleLoadingStatus) {
			//unload previously loaded module
			STPcudaCheckErr(cuModuleUnload(this->GeneratorProgram));
		}
		STPLinkerInformation::STPDataJitOption& moduleOption = linker_info.ModuleOption;
		STPcudaCheckErr(cuModuleLoadDataEx(&this->GeneratorProgram, program_cubin,
			static_cast<int>(moduleOption.OptionFlag.size()), moduleOption.OptionFlag.data(), moduleOption.OptionValue.data()));
		this->ModuleLoadingStatus = true;
	}
	catch (...) {
		exptr = current_exception();
	}

	//delete the link regardlessly
	STPcudaCheckErr(cuLinkDestroy(linker));
	if (exptr) {
		//throw any module exception
		rethrow_exception(exptr);
	}
	
}

const STPDiversityGeneratorRTC::STPLoweredName& STPDiversityGeneratorRTC::retrieveSourceLoweredName(string source_name) const {
	auto name_expression = this->CompilationNameDatabase.find(source_name);
	if (name_expression == this->CompilationNameDatabase.end()) {
		throw STPException::STPMemoryError((string(__FILE__) + "::" + string(__FUNCTION__) + "\nSource name cannot be found in source database.").c_str());
	}
	return name_expression->second;
}

CUmodule STPDiversityGeneratorRTC::getGeneratorModule() const {
	return this->GeneratorProgram;
}

#pragma warning(default:26812)