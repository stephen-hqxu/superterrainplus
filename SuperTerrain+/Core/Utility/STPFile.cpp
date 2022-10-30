#include <SuperTerrain+/Utility/STPFile.h>

//Error
#include <SuperTerrain+/Exception/STPSerialisationError.h>

//IO
#include <fstream>
#include <sstream>

using std::string;
using std::ifstream;
using std::ostringstream;
using std::istreambuf_iterator;

using namespace SuperTerrainPlus;

string STPFile::read(const char* filename) {
	using std::ios;
	//open the file
	ifstream fileIO(filename, ios::in);
	if (!fileIO.good()) {
		//cannot open the file
		ostringstream msg;
		msg << "File \'" << filename << "\' cannot be opened" << std::endl;
		throw STPException::STPSerialisationError(msg.str().c_str());
	}

	//read all lines
	string content;
	//reserve space for string output
	fileIO.seekg(0, ios::end);
	content.reserve(fileIO.tellg());
	fileIO.seekg(0, ios::beg);
	//copy to output
	content.assign(istreambuf_iterator<char>(fileIO), istreambuf_iterator<char>());

	return content;
}