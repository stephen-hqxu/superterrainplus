#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPDatabaseError.h>

//Database
#include <SuperTerrain+/Utility/STPSQLite.h>
//System
#include <string>
#include <string_view>
#include <algorithm>
#include <array>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::move;
using std::make_tuple;
using std::make_pair;
using std::make_unique;
using std::string;
using std::string_view;
using std::unique_ptr;

using glm::uvec2;

struct STPTextureDatabase::STPTextureDatabaseImpl {
private:

	inline static unsigned int InstanceCounter = 0u;

	//we use sqlite3 as the database implementation, this is the SQL database connection being setup
	sqlite3* SQL;

	/**
	 * @brief STPStmtFinaliser finalises sqlite3_stmt
	*/
	struct STPStmtFinaliser {
	public:

		inline void operator()(sqlite3_stmt* stmt) const {
			STPsqliteCheckErr(sqlite3_finalize(stmt));
		}

	};

	//An auto-managed SQL prepare statement
	typedef unique_ptr<sqlite3_stmt, STPStmtFinaliser> STPSmartStmt;
	//All reusable prepare statements
	std::array<STPSmartStmt, 4> Statement;

public:

	//prepare statement indexing
	typedef unsigned int STPStatementID;
	constexpr static STPStatementID AddAltitude = 0u;
	constexpr static STPStatementID AddGradient = 1u;
	constexpr static STPStatementID AddGroup = 2u;
	constexpr static STPStatementID AddTextureData = 3u;

	/**
	 * @brief Init STPTextureDatabaseImpl, setup database connection
	*/
	STPTextureDatabaseImpl() {
		const string filename = "STPTextureDatabase_" + (STPTextureDatabaseImpl::InstanceCounter++);
		//open database connection
		STPsqliteCheckErr(sqlite3_open_v2(filename.c_str(), &this->SQL,
			SQLITE_OPEN_MEMORY | SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOFOLLOW, nullptr));
		//enforce foreign key constraints
		int fk;
		STPsqliteCheckErr(sqlite3_db_config(this->SQL, SQLITE_DBCONFIG_ENABLE_FKEY, 1, &fk));
		if (fk != 1) {
			//usually this should not happen, unless sqlite is bugged
			throw STPException::STPDatabaseError("Foreign key constraints cannot be turned on");
		}
	}

	STPTextureDatabaseImpl(const STPTextureDatabaseImpl&) = delete;

	STPTextureDatabaseImpl(STPTextureDatabaseImpl&&) = delete;

	STPTextureDatabaseImpl& operator=(const STPTextureDatabaseImpl&) = delete;

	STPTextureDatabaseImpl& operator=(STPTextureDatabaseImpl&&) = delete;

	~STPTextureDatabaseImpl() {
		//close the database connection
		STPsqliteCheckErr(sqlite3_close_v2(this->SQL));
	}

	/**
	 * @brief Execute semicolon-separated SQLs to create tables
	 * @param sql SQL statements, each create table statements must be separated by semicolon
	*/
	void createFromSchema(const string_view& sql) {
		char* err_msg;
		try {
			//we don't need callback function since create table does not return anything
			STPsqliteCheckErr(sqlite3_exec(this->SQL, sql.data(), nullptr, nullptr, &err_msg));
		}
		catch (const STPException::STPDatabaseError& dbe) {
			//concate error message and throw a new one
			//usually exception should not be thrown since schema is created by developer not client
			const string compound = string(dbe.what()) + "\nMessage: " + err_msg;
			sqlite3_free(err_msg);
			throw STPException::STPDatabaseError(compound.c_str());
		}
	}

	/**
	 * @brief Get the prepared statement managed by the database.
	 * It effectively creates and reuse the statement.
	 * @param sid The ID of the statement to be retrieved, if has been previously created; or assigned, if needs to be created
	 * @param sql The SQL query used to create. This argument is ignored if statement has been created previously.
	 * @return The pointer to the statement
	*/
	sqlite3_stmt* getStmt(STPStatementID sid, const string_view& sql) {
		STPSmartStmt& stmt = this->Statement[sid];

		if (!stmt) {
			//if statement is not previously created, create one now
			sqlite3_stmt* newStmt;
			STPsqliteCheckErr(sqlite3_prepare_v3(this->SQL, sql.data(), sql.size(), SQLITE_PREPARE_PERSISTENT, &newStmt, nullptr));
			stmt = STPSmartStmt(newStmt);
			//return the newly created and managed statement object
			return newStmt;
		}
		//get the statement
		return stmt.get();
	}

	/**
	 * @brief Execute a prepare statement once, throw exception if execution fails
	 * @param stmt The pointer to prepare statement to be executed
	 * @param operation_msg The message to be prefixed to the exeception message
	*/
	inline void execStmt(sqlite3_stmt* stmt, const char* operation_msg) {
		if (const int err_code = sqlite3_step(stmt);
			err_code != SQLITE_DONE) {
			const string err_msg = string(operation_msg) + "\nMessage: " + sqlite3_errstr(err_code);
			throw STPException::STPDatabaseError(err_msg.c_str());
		}
	}

};

STPTextureDatabase::STPTextureSplatBuilder::STPTextureSplatBuilder(STPTextureDatabase::STPTextureDatabaseImpl* database) : Database(database) {
	//setup database schema for splat builder
	static constexpr string_view TextureSplatSchema =
		"CREATE TABLE AltitudeStructure("
			"ASID INT NOT NULL,"
			"Sample INT NOT NULL,"
			"UpperBound FLOAT NOT NULL,"
			"TID INT NOT NULL,"
			"PRIMARY KEY(ASID), FOREIGN KEY(TID) REFERENCES Texture(TID) ON DELETE CASCADE"
		");"
		"CREATE TABLE GradientStructure("
			"GSID INT NOT NULL,"
			"Sample INT NOT NULL,"
			"minGradient FLOAT NOT NULL,"
			"maxGradient FLOAT NOT NULL,"
			"LowerBound FLOAT NOT NULL,"
			"UpperBound FLOAT NOT NULL,"
			"TID INT NOT NULL,"
			"PRIMARY KEY(GSID), FOREIGN KEY(TID) REFERENCES Texture(TID) ON DELETE CASCADE"
		");";

	//create table
	this->Database->createFromSchema(TextureSplatSchema);
}

void STPTextureDatabase::STPTextureSplatBuilder::addAltitude(Sample sample, float upperBound, STPTextureInformation::STPTextureID texture_id) {
	static constexpr string_view AddAltitude = 
		"INSERT INTO AltitudeStructure (ASID, Sample, UpperBound, TID) VALUES(?, ?, ?, ?)";
	sqlite3_stmt* altitude_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddAltitude, AddAltitude);

	STPsqliteCheckErr(sqlite3_reset(altitude_stmt));
	//insert new altitude configuration into altitude table
	STPsqliteCheckErr(sqlite3_bind_int(altitude_stmt, 1, static_cast<int>(STPTextureDatabase::GeneralIDAccumulator++)));
	STPsqliteCheckErr(sqlite3_bind_int(altitude_stmt, 2, static_cast<int>(sample)));
	STPsqliteCheckErr(sqlite3_bind_double(altitude_stmt, 3, static_cast<double>(upperBound)));
	STPsqliteCheckErr(sqlite3_bind_int(altitude_stmt, 4, static_cast<int>(texture_id)));
	//execute
	this->Database->execStmt(altitude_stmt, "Altitude cannot be added");
}

void STPTextureDatabase::STPTextureSplatBuilder::addGradient
	(Sample sample, float minGradient, float maxGradient, float lowerBound, float upperBound, STPTextureInformation::STPTextureID texture_id) {
	static constexpr string_view AddGradient =
		"INSERT INTO GradientStructure (GSID, Sample, minGradient, maxGradient, LowerBound, UpperBound, TID) VALUES(?, ?, ?, ?, ?, ?, ?);";
	sqlite3_stmt* gradient_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddGradient, AddGradient);

	STPsqliteCheckErr(sqlite3_reset(gradient_stmt));
	//insert new gradient configuration into gradient table
	STPsqliteCheckErr(sqlite3_bind_int(gradient_stmt, 1, static_cast<int>(STPTextureDatabase::GeneralIDAccumulator++)));
	STPsqliteCheckErr(sqlite3_bind_int(gradient_stmt, 2, static_cast<int>(sample)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 3, static_cast<double>(minGradient)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 4, static_cast<double>(maxGradient)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 5, static_cast<double>(lowerBound)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 6, static_cast<double>(upperBound)));
	STPsqliteCheckErr(sqlite3_bind_int(gradient_stmt, 7, static_cast<int>(texture_id)));

	this->Database->execStmt(gradient_stmt, "Gradient cannot be added");
}

STPTextureDatabase::STPTextureDatabase() : Database(make_unique<STPTextureDatabaseImpl>()), SplatBuilder(this->Database.get()) {
	//setup database schema for texture database
	//no need to drop table since database is freshly created
	static constexpr string_view TextureDatabaseSchema =
		"CREATE TABLE Texture("
			"TID INT NOT NULL,"
			"TextureType TINYINT NOT NULL,"
			"TGID INT NOT NULL,"
			//store temp-pointer (yes a pointer, not the actual texture data) to texture data
			"TextureData BLOB NOT NULL,"
			"PRIMARY KEY(TID, TextureType) FOREIGN KEY(TGID) REFERENCES TextureGroup(TGID) ON DELETE CASCADE"
		");"
		"CREATE TABLE TextureGroup("
			"TGID INT NOT NULL,"
			//STPTextureDescription
			"TextureDescription BLOB NOT NULL,"
			"PRIMARY KEY(TGID)"
		");";

	//create table
	//if no error occurs, error message will be set to nullptr
	//normally there should be no exception
	this->Database->createFromSchema(TextureDatabaseSchema);
}

STPTextureDatabase::~STPTextureDatabase() = default;

STPTextureInformation::STPTextureGroupID STPTextureDatabase::addGroup(const STPTextureDescription& desc) {
	static constexpr string_view AddGroup =
		"INSERT INTO TextureGroup (TGID, TextureDescription) VALUES(?, ?);";
	sqlite3_stmt* group_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddGroup, AddGroup);

	STPsqliteCheckErr(sqlite3_reset(group_stmt));
	//insert texture desc as a binary
	STPsqliteCheckErr(sqlite3_bind_int(group_stmt, 1, static_cast<int>(++STPTextureDatabase::GroupIDAccumulator)));
	//we tell sqlite to copy the object and it will manage its lifetime for us
	STPsqliteCheckErr(sqlite3_bind_blob(group_stmt, 2, &desc, sizeof(desc), SQLITE_TRANSIENT));

	this->Database->execStmt(group_stmt, "Group cannot be added due to database error");
	return STPTextureDatabase::GroupIDAccumulator;
}

STPTextureInformation::STPTextureID STPTextureDatabase::addTexture() {
	return STPTextureDatabase::TextureIDAccumulator++;
}

void STPTextureDatabase::addTextureData
	(STPTextureInformation::STPTextureID texture_id, STPTextureType type, STPTextureInformation::STPTextureGroupID group_id, const void* texture_data) {
	static constexpr string_view AddTextureData =
		"INSERT INTO Texture (TID, TextureType, TGID, TextureData) VALUES(?, ?, ?, ?);";
	sqlite3_stmt* texture_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddTextureData, AddTextureData);

	STPsqliteCheckErr(sqlite3_reset(texture_stmt));
	//set data
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 1, static_cast<int>(texture_id)));
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 2, static_cast<int>(type)));
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 3, static_cast<int>(group_id)));
	//send the pointer as a blob
	//here we have told user to manage the lifetime of texture for us, so sqlite doesn't need to worry about that
	//never assume the size of a pointer
	STPsqliteCheckErr(sqlite3_bind_blob(texture_stmt, 4, texture_data, sizeof(texture_data), SQLITE_STATIC));

	this->Database->execStmt(texture_stmt, "Texture data cannot be added for the specified texture ID and type");
}