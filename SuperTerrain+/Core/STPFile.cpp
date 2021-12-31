#include <SuperTerrain+/Utility/STPFile.h>

//Error
#include <SuperTerrain+/Exception/STPSerialisationError.h>

//IO
#include <fstream>
#include <sstream>
#include <streambuf>

using std::string;
using std::ifstream;
using std::stringstream;
using std::istreambuf_iterator;

using namespace SuperTerrainPlus;

STPFile::STPFile(const char* filename) {
	using std::ios;
	//open the file
	ifstream fileIO(filename, ios::in);
	if (!fileIO.good()) {
		//cannot open the file
		stringstream msg;
		msg << "File \'" << filename << "\' cannot be opened" << std::endl;
		throw STPException::STPSerialisationError(msg.str().c_str());
	}

	//read all lines
	//reserve space for string output
	fileIO.seekg(0, ios::end);
	this->Content.reserve(fileIO.tellg());
	fileIO.seekg(0, ios::beg);
	//copy to output
	this->Content.assign(istreambuf_iterator<char>(fileIO), istreambuf_iterator<char>());
}

const string& STPFile::operator*() const {
	return this->Content;
}