#include <SuperTerrain+/GPGPU/STPDeviceRuntimeBinary.h>

//Error
#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//System
#include <algorithm>
#include <sstream>
#include <exception>

using std::vector;
using std::string;
using std::ostringstream;
using std::exception_ptr;

using std::make_unique;
using std::endl;

using namespace SuperTerrainPlus;

STPDeviceRuntimeBinary::STPSourceInformation::STPSourceArgument&
	STPDeviceRuntimeBinary::STPSourceInformation::STPSourceArgument::operator[](const string& arg) {
	this->StringArgument.emplace_back(arg);
	return *this;
}

void STPDeviceRuntimeBinary::STPProgramDeleter::operator()(nvrtcProgram program) const {
	STP_CHECK_CUDA(nvrtcDestroyProgram(&program));
}

STPDeviceRuntimeBinary::STPCompilationOutput STPDeviceRuntimeBinary::compile(
	string&& source_name, const string& source_code,
	const STPSourceInformation& source_info, const STPExternalHeaderSource& external_header) {
	STPCompilationOutput output;
	auto& [program_object, output_log, output_name] = output;
	auto& [program_name, managed_program] = program_object;
	program_name = std::move(source_name);

	const auto& [src_option, src_name_expr, src_external_header] = source_info;

	//convert array of strings to array of char pointers
	vector<const char*> raw_header_name, raw_header_code;
	//search each external headers from the provided argument
	for (const auto& required_header_name : src_external_header.StringArgument) {
		auto it = external_header.find(required_header_name);
		if (it == external_header.cend()) {
			//cannot find the source of this header
			ostringstream err;
			err << "External header '" << required_header_name << "\' is required for source '"
				<< program_name << "\' but its definition is not found." << endl;

			throw STPException::STPCompilationError(err.str().c_str());
		}

		const auto& [db_header_name, db_header_source] = *it;
		raw_header_name.emplace_back(db_header_name.c_str());
		raw_header_code.emplace_back(db_header_source.c_str());
	}

	//external_header and external_header_code should have the same size
	//create a new program
	nvrtcProgram program;
	STP_CHECK_CUDA(nvrtcCreateProgram(&program, source_code.c_str(), program_name.c_str(),
		static_cast<int>(raw_header_name.size()), raw_header_code.data(), raw_header_name.data()));
	managed_program = STPSmartProgram(program);
	//compile the new program
	const auto& name_expr_arg = src_name_expr.StringArgument;
	exception_ptr exptr;
	try {
		//extract raw string options
		const auto& option_arg = src_option.StringArgument;
		vector<const char*> raw_option;
		raw_option.resize(option_arg.size());
		std::transform(option_arg.cbegin(), option_arg.cend(), raw_option.begin(), [](const auto& str) { return str.c_str(); });

		//add name expression
		std::for_each(name_expr_arg.cbegin(), name_expr_arg.cend(),
			[program](const auto& str) { STP_CHECK_CUDA(nvrtcAddNameExpression(program, str.c_str())); });

		//compile
		STP_CHECK_CUDA(nvrtcCompileProgram(program, static_cast<int>(raw_option.size()), raw_option.data()));

	} catch (...) {
		//restore the exception if any, because we want to get the log
		exptr = std::current_exception();
	}

	//logging
	size_t logSize;
	STP_CHECK_CUDA(nvrtcGetProgramLogSize(program, &logSize));
	output_log.resize(logSize);
	STP_CHECK_CUDA(nvrtcGetProgramLog(program, output_log.data()));

	if (exptr) {
		throw STPException::STPCompilationError(output_log.c_str());
	}
	//if no error appears grab the lowered name expression from compiled program
	//declare a new lowered name storage, so if any errors appears later it will not affect the storage in the class.
	output_name.reserve(name_expr_arg.size());
	for (const auto& expr : name_expr_arg) {
		const char* lowered_name;
		//we expect every name added previously are valid
		STP_CHECK_CUDA(nvrtcGetLoweredName(program, expr.c_str(), &lowered_name));

		//add it to the output
		output_name.emplace(expr, lowered_name);
	}

	return output;
}

STPDeviceRuntimeBinary::STPProgramData STPDeviceRuntimeBinary::readNVVM(nvrtcProgram program) {
	STPProgramData data;
	auto& [nvvm, nvvmSize] = data;

	STP_CHECK_CUDA(nvrtcGetNVVMSize(program, &nvvmSize));
	nvvm = make_unique<char[]>(nvvmSize);
	STP_CHECK_CUDA(nvrtcGetNVVM(program, nvvm.get()));

	return data;
}

STPDeviceRuntimeBinary::STPProgramData STPDeviceRuntimeBinary::readPTX(nvrtcProgram program) {
	STPProgramData data;
	auto& [ptx, ptxSize] = data;

	STP_CHECK_CUDA(nvrtcGetPTXSize(program, &ptxSize));
	ptx = make_unique<char[]>(ptxSize);
	STP_CHECK_CUDA(nvrtcGetPTX(program, ptx.get()));
	
	return data;
}

STPDeviceRuntimeBinary::STPProgramData STPDeviceRuntimeBinary::readCUBIN(nvrtcProgram program) {
	STPProgramData data;
	auto& [cubin, cubinSize] = data;

	STP_CHECK_CUDA(nvrtcGetCUBINSize(program, &cubinSize));
	cubin = make_unique<char[]>(cubinSize);
	STP_CHECK_CUDA(nvrtcGetCUBIN(program, cubin.get()));

	return data;
}