#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPDatabaseError.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>

//Database
#include <SuperTerrain+/STPSQLite.h>
//System
#include <string>
#include <algorithm>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::make_pair;
using std::make_unique;
using std::string;
using std::string_view;
using std::unique_ptr;

using glm::uvec2;

class STPTextureDatabase::STPTextureDatabaseImpl {
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

public:

	//An auto-managed SQL prepare statement
	typedef unique_ptr<sqlite3_stmt, STPStmtFinaliser> STPSmartStmt;

private:

	//All reusable prepare statements
	//we try to reuse statements that are supposed to be called very frequently
	STPSmartStmt Statement[8];

public:

	//prepare statement indexing
	typedef unsigned int STPStatementID;
	constexpr static STPStatementID
		AddAltitude = 0u,
		AddGradient = 1u,
		AddGroup = 2u,
		RemoveGroup = 3u,
		GetGroup = 4u,
		AddTexture = 5u,
		RemoveTexture = 6u,
		AddMap = 7u;

	/**
	 * @brief Init STPTextureDatabaseImpl, setup database connection
	*/
	STPTextureDatabaseImpl() {
		const string filename = "STPTextureDatabase_" + (STPTextureDatabaseImpl::InstanceCounter++);
		//open database connection
		STPsqliteCheckErr(sqlite3_open_v2(filename.c_str(), &this->SQL,
			SQLITE_OPEN_MEMORY | SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOFOLLOW, nullptr));

		auto configure = [SQL = this->SQL](int opt, int val) -> bool {
			int report;
			STPsqliteCheckErr(sqlite3_db_config(SQL, opt, val, &report));
			//usually it should always be true, unless sqlite is bugged
			return report == val;
		};
		//enforce foreign key constraints
		if (!configure(SQLITE_DBCONFIG_ENABLE_FKEY, 1)) {
			throw STPException::STPDatabaseError("Foreign key constraints cannot be enforced");
		}
		//enable view
		if (!configure(SQLITE_DBCONFIG_ENABLE_VIEW, 1)) {
			throw STPException::STPDatabaseError("View cannot be enabled");
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
	 * @brief Reset stmt back to its initial state
	 * @param stmt The statement to be reset
	*/
	static void resetStmt(sqlite3_stmt* stmt) {
		STPsqliteCheckErr(sqlite3_reset(stmt));
		STPsqliteCheckErr(sqlite3_clear_bindings(stmt));
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
	 * @brief Create a SQL prepare statement object, the statement is auto-managed for garbage collection
	 * @param sql The SQL query used to create.
	 * @param flag Optional prepare statement flag
	 * @return The smart prepared statement
	*/
	STPSmartStmt createStmt(const string_view& sql, unsigned int flag = 0u) {
		sqlite3_stmt* newStmt;
		STPsqliteCheckErr(sqlite3_prepare_v3(this->SQL, sql.data(), sql.size(), flag, &newStmt, nullptr));
		return STPSmartStmt(newStmt);
	}

	/**
	 * @brief Get the prepared statement managed by the database.
	 * It effectively creates and reuse the statement.
	 * @param sid The ID of the statement to be retrieved, if has been previously created; or assigned, if needs to be created
	 * @param sql The SQL query used to create. This argument is ignored if statement has been created previously.
	 * @return The pointer to the statement. The statement will be ready to be used.
	*/
	sqlite3_stmt* getStmt(STPStatementID sid, const string_view& sql) {
		STPSmartStmt& stmt = this->Statement[sid];

		if (!stmt) {
			//if statement is not previously created, create one now
			stmt = this->createStmt(sql, SQLITE_PREPARE_PERSISTENT);
			//return the newly created and managed statement object
			return stmt.get();
		}
		//get the statement
		sqlite3_stmt* currStmt = stmt.get();
		//reset the statement before returning, since it may contain state from the previous call
		STPTextureDatabaseImpl::resetStmt(currStmt);
		return currStmt;
	}

	/**
	 * @brief Execute a prepare statement once, throw exception if execution fails
	 * @param stmt The pointer to prepare statement to be executed
	 * @return True if the statement has another row ready, in other words, step status == SQLITE_ROW not SQLITE_DONE, and no error occurs
	*/
	inline bool execStmt(sqlite3_stmt* stmt) {
		const int err_code = sqlite3_step(stmt);
		if (err_code != SQLITE_ROW && err_code != SQLITE_DONE) {
			//error occurs
			//Exploiting API behaviour: SQLite will throw exception at reset
			STPTextureDatabaseImpl::resetStmt(stmt);
		}
		return err_code == SQLITE_ROW;
	}

	/**
	 * @brief A helper function for queries that contain a single output, for example COUNT()
	 * @param sql The query that will return a single output
	 * @return The only output from the query
	*/
	size_t getSingleOutput(const string_view& sql) {
		STPTextureDatabaseImpl::STPSmartStmt smart_stmt = this->createStmt(sql);
		sqlite3_stmt* const texture_stmt = smart_stmt.get();

		//no data to be bound, exec
		if (!this->execStmt(texture_stmt)) {
			//these types of query doesn't have any condition so does not rely on user input.
			//If fails, there must be something wrong with program itself.
			throw STPException::STPDatabaseError("Query that supposed to produce a result does not. "
				"This error is caused by the program itself, please contact the engine maintainer.");
		}
		//we only expect a single row and column
		return sqlite3_column_int(texture_stmt, 0);
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
			"PRIMARY KEY(GSID), FOREIGN KEY(TID) REFERENCES Texture(TID) ON DELETE CASCADE,"
			"CHECK(minGradient <= maxGradient AND LowerBound <= UpperBound)"
		");"
		//A view that holds all maps with texture ID being referenced by any of the splat structure
		"CREATE VIEW ValidMap AS "
			//there is no ANY operator in SQLite, we can do it with IN
			"SELECT * FROM Map WHERE TID IN("
				"SELECT TID FROM AltitudeStructure "
				"UNION "
				"SELECT TID FROM GradientStructure"
			");";

	//create table
	this->Database->createFromSchema(TextureSplatSchema);
}

STPTextureDatabase::STPTextureSplatBuilder::STPTextureSplatBuilder(STPTextureSplatBuilder&&) noexcept = default;

STPTextureDatabase::STPTextureSplatBuilder& STPTextureDatabase::STPTextureSplatBuilder::operator=(STPTextureSplatBuilder&&) noexcept = default;

void STPTextureDatabase::STPTextureSplatBuilder::addAltitude(Sample sample, float upperBound, STPTextureInformation::STPTextureID texture_id) {
	static constexpr string_view AddAltitude = 
		"INSERT INTO AltitudeStructure (ASID, Sample, UpperBound, TID) VALUES(?, ?, ?, ?)";
	sqlite3_stmt* const altitude_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddAltitude, AddAltitude);

	//insert new altitude configuration into altitude table
	STPsqliteCheckErr(sqlite3_bind_int(altitude_stmt, 1, static_cast<int>(STPTextureDatabase::GeneralIDAccumulator++)));
	STPsqliteCheckErr(sqlite3_bind_int(altitude_stmt, 2, static_cast<int>(sample)));
	STPsqliteCheckErr(sqlite3_bind_double(altitude_stmt, 3, static_cast<double>(upperBound)));
	STPsqliteCheckErr(sqlite3_bind_int(altitude_stmt, 4, static_cast<int>(texture_id)));
	//execute
	this->Database->execStmt(altitude_stmt);
}

size_t STPTextureDatabase::STPTextureSplatBuilder::altitudeSize() const {
	static constexpr string_view GetAltitudeCount = 
		"SELECT COUNT(ASID) FROM AltitudeStructure;";
	
	return this->Database->getSingleOutput(GetAltitudeCount);
}

void STPTextureDatabase::STPTextureSplatBuilder::addGradient
	(Sample sample, float minGradient, float maxGradient, float lowerBound, float upperBound, STPTextureInformation::STPTextureID texture_id) {
	static constexpr string_view AddGradient =
		"INSERT INTO GradientStructure (GSID, Sample, minGradient, maxGradient, LowerBound, UpperBound, TID) VALUES(?, ?, ?, ?, ?, ?, ?);";
	sqlite3_stmt* const gradient_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddGradient, AddGradient);

	//insert new gradient configuration into gradient table
	STPsqliteCheckErr(sqlite3_bind_int(gradient_stmt, 1, static_cast<int>(STPTextureDatabase::GeneralIDAccumulator++)));
	STPsqliteCheckErr(sqlite3_bind_int(gradient_stmt, 2, static_cast<int>(sample)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 3, static_cast<double>(minGradient)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 4, static_cast<double>(maxGradient)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 5, static_cast<double>(lowerBound)));
	STPsqliteCheckErr(sqlite3_bind_double(gradient_stmt, 6, static_cast<double>(upperBound)));
	STPsqliteCheckErr(sqlite3_bind_int(gradient_stmt, 7, static_cast<int>(texture_id)));

	this->Database->execStmt(gradient_stmt);
}

size_t STPTextureDatabase::STPTextureSplatBuilder::gradientSize() const {
	static constexpr string_view GetGradientCount = 
		"SELECT COUNT(GSID) FROM GradientStructure;";
	
	return this->Database->getSingleOutput(GetGradientCount);
}

STPTextureDatabase::STPDatabaseView::STPDatabaseView(const STPTextureDatabase& db) : Database(db), Impl(db.Database.get()), SplatBuilder(db.SplatBuilder) {

}

STPTextureDatabase::STPDatabaseView::STPAltitudeRecord STPTextureDatabase::STPDatabaseView::getAltitudes() const {
	static constexpr string_view GetAltitude =
		"SELECT Sample, UpperBound, TID FROM AltitudeStructure ORDER BY Sample ASC, UpperBound ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_altitude_stmt = this->Impl->createStmt(GetAltitude);
	sqlite3_stmt* const altitude_stmt = smart_altitude_stmt.get();
	STPAltitudeRecord altitude_rec;
	//preallocate memory
	altitude_rec.reserve(this->SplatBuilder.altitudeSize());

	//structure the altitude records and add them into the vector
	while (this->Impl->execStmt(altitude_stmt)) {
		STPTextureInformation::STPAltitudeNode& newAlt =
			altitude_rec.emplace_back(static_cast<Sample>(sqlite3_column_int(altitude_stmt, 0)), STPTextureInformation::STPAltitudeNode()).second;
		newAlt.UpperBound = static_cast<float>(sqlite3_column_double(altitude_stmt, 1));
		newAlt.Reference.DatabaseKey = static_cast<STPTextureInformation::STPTextureID>(sqlite3_column_int(altitude_stmt, 2));
	}

	return altitude_rec;
}

STPTextureDatabase::STPDatabaseView::STPGradientRecord STPTextureDatabase::STPDatabaseView::getGradients() const {
	static constexpr string_view GetGradient =
		"SELECT Sample, minGradient, maxGradient, LowerBound, UpperBound, TID FROM GradientStructure ORDER BY Sample ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_gradient_stmt = this->Impl->createStmt(GetGradient);
	sqlite3_stmt* const gradient_stmt = smart_gradient_stmt.get();
	STPGradientRecord gradient_rec;
	//preallocate memory
	gradient_rec.reserve(this->SplatBuilder.gradientSize());

	//structure then add
	while (this->Impl->execStmt(gradient_stmt)) {
		STPTextureInformation::STPGradientNode& newGra =
			gradient_rec.emplace_back(static_cast<Sample>(sqlite3_column_int(gradient_stmt, 0)), STPTextureInformation::STPGradientNode()).second;
		newGra.minGradient = static_cast<float>(sqlite3_column_double(gradient_stmt, 1));
		newGra.maxGradient = static_cast<float>(sqlite3_column_double(gradient_stmt, 2));
		newGra.LowerBound = static_cast<float>(sqlite3_column_double(gradient_stmt, 3));
		newGra.UpperBound = static_cast<float>(sqlite3_column_double(gradient_stmt, 4));
		newGra.Reference.DatabaseKey = static_cast<STPTextureInformation::STPTextureID>(sqlite3_column_int(gradient_stmt, 5));
	}

	return gradient_rec;
}

STPTextureDatabase::STPDatabaseView::STPSampleRecord STPTextureDatabase::STPDatabaseView::getValidSample(unsigned int hint) const {
	//get sample ID that appears in both splat structures
	static constexpr string_view GetAffectedSample =
		//sqlite does not support full outer join, we can union two left outer joins to emulate the behaviour
		"SELECT P.Sample, IFNULL(A.AltCount, 0), IFNULL(G.GraCount, 0) FROM ("
			"SELECT Sample FROM AltitudeStructure "
			"UNION "
			"SELECT Sample FROM GradientStructure"
		") P "
		"LEFT OUTER JOIN (SELECT Sample, Count(Sample) AS AltCount FROM AltitudeStructure GROUP BY Sample) A ON A.Sample = P.Sample "
		"LEFT OUTER JOIN (SELECT Sample, Count(Sample) AS GraCount FROM GradientStructure GROUP BY Sample) G ON G.Sample = P.Sample "
		"ORDER BY P.Sample ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_affected_stmt = this->Impl->createStmt(GetAffectedSample);
	sqlite3_stmt* const affected_stmt = smart_affected_stmt.get();
	STPSampleRecord sample_rec;
	sample_rec.reserve(hint);

	//insert into sample array
	while (this->Impl->execStmt(affected_stmt)) {
		sample_rec.emplace_back(
			static_cast<Sample>(sqlite3_column_int(affected_stmt, 0)),
			static_cast<size_t>(sqlite3_column_int(affected_stmt, 1)),
			static_cast<size_t>(sqlite3_column_int(affected_stmt, 2))
		);
	}

	//the array will probably be read-only once returned, to avoid wasting memory we shrink it
	sample_rec.shrink_to_fit();
	return sample_rec;
}

STPTextureDatabase::STPDatabaseView::STPGroupRecord STPTextureDatabase::STPDatabaseView::getValidGroup() const {
	//we are only interested in group that is used by some texture data, for groups that are not referenced, ignore them.
	static constexpr string_view GetAllGroup =
		"SELECT TG.TGID, COUNT(VM.TGID), TG.Description FROM TextureGroup TG "
			"INNER JOIN ValidMap VM ON TG.TGID = VM.TGID "
		"GROUP BY VM.TGID ORDER BY TG.TGID ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_group_stmt = this->Impl->createStmt(GetAllGroup);
	sqlite3_stmt* const group_stmt = smart_group_stmt.get();
	STPGroupRecord group_rc;
	//estimate the size, the number of valid group must be less than or equal to the total number of group
	group_rc.reserve(this->Database.groupSize());

	//loop through all group data and emplace them into a strutured array
	while (this->Impl->execStmt(group_stmt)) {
		group_rc.emplace_back(
			static_cast<STPTextureInformation::STPTextureGroupID>(sqlite3_column_int(group_stmt, 0)),
			static_cast<size_t>(sqlite3_column_int(group_stmt, 1)),
			*reinterpret_cast<const STPTextureDescription*>(sqlite3_column_blob(group_stmt, 2))
		);
	}

	//clear up unused data
	group_rc.shrink_to_fit();
	return group_rc;
}

STPTextureDatabase::STPDatabaseView::STPTextureRecord STPTextureDatabase::STPDatabaseView::getValidTexture() const {
	static constexpr string_view GetAllTexture =
		"SELECT DISTINCT TID FROM ValidMap ORDER BY TID ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_texture_stmt = this->Impl->createStmt(GetAllTexture);
	sqlite3_stmt* const texture_stmt = smart_texture_stmt.get();
	STPTextureRecord texture_rec;
	//The number of valid texture ID must be less than or equal to the total number of registered texture
	texture_rec.reserve(this->Database.textureSize());

	while (this->Impl->execStmt(texture_stmt)) {
		//add texture ID into the array
		texture_rec.emplace_back(static_cast<STPTextureInformation::STPTextureID>(sqlite3_column_int(texture_stmt, 0)));
	}

	//clear up
	texture_rec.shrink_to_fit();
	return texture_rec;
}

STPTextureDatabase::STPDatabaseView::STPMapRecord STPTextureDatabase::STPDatabaseView::getValidMap() const {
	static constexpr string_view GetAllTextureData =
		"SELECT TGID, TID, Type, Data FROM ValidMap ORDER BY TGID ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_texture_stmt = this->Impl->createStmt(GetAllTextureData);
	sqlite3_stmt* const texture_stmt = smart_texture_stmt.get();
	STPMapRecord texture_rec;
	//reserve data, since we are retrieving all texture data, the size is exact
	texture_rec.reserve(this->Database.mapSize());

	while (this->Impl->execStmt(texture_stmt)) {
		texture_rec.emplace_back(
			static_cast<STPTextureInformation::STPTextureGroupID>(sqlite3_column_int(texture_stmt, 0)),
			static_cast<STPTextureInformation::STPTextureID>(sqlite3_column_int(texture_stmt, 1)),
			static_cast<STPTextureType>(sqlite3_column_int(texture_stmt, 2)),
			//sqlite will allocate new space for the blob result
			//because we are efftively storing a pointer in the database previously, so the allocated memory is a pointer to pointer
			*reinterpret_cast<const void* const*>(sqlite3_column_blob(texture_stmt, 3))
		);
	}

	//no need to clean up, because the reserved size is exact
	return texture_rec;
}

STPTextureDatabase::STPDatabaseView::STPTextureTypeRecord STPTextureDatabase::STPDatabaseView::getValidMapType(unsigned int hint) const {
	if (hint > static_cast<std::underlying_type_t<STPTextureType>>(STPTextureType::TypeCount)) {
		throw STPException::STPBadNumericRange("Hint is larger than the number of possible type");
	}

	static constexpr string_view GetAllType = 
		"SELECT DISTINCT Type FROM ValidMap ORDER BY Type ASC;";
	STPTextureDatabaseImpl::STPSmartStmt smart_texture_stmt = this->Impl->createStmt(GetAllType);
	sqlite3_stmt* const texture_stmt = smart_texture_stmt.get();
	STPTextureTypeRecord type_rec;
	//reserve with hint
	type_rec.reserve(hint);

	while (this->Impl->execStmt(texture_stmt)) {
		type_rec.emplace_back(static_cast<STPTextureType>(sqlite3_column_int(texture_stmt, 0)));
	}

	//clear up
	type_rec.shrink_to_fit();
	return type_rec;
}

STPTextureDatabase::STPTextureDatabase() : Database(make_unique<STPTextureDatabaseImpl>()), SplatBuilder(this->Database.get()) {
	//setup database schema for texture database
	//no need to drop table since database is freshly created
	static constexpr string_view TextureDatabaseSchema =
		//A texture contains maps of different types, for example a grass texture may have albedo, normal and specular map
		"CREATE TABLE Texture("
			"TID INT NOT NULL,"
			"Name VARCHAR(10),"
			"PRIMARY KEY(TID)"
		");"
		"CREATE TABLE TextureGroup("
			"TGID INT NOT NULL,"
			//STPTextureDescription
			"Description BLOB NOT NULL,"
			"PRIMARY KEY(TGID)"
		");"
		"CREATE TABLE Map("
			"MID INT NOT NULL,"
			"Type TINYINT NOT NULL,"
			"TGID INT NOT NULL,"
			//store temp-pointer (yes a pointer, not the actual texture data) to texture data
			"Data BLOB NOT NULL,"
			"TID INT NOT NULL,"
			"PRIMARY KEY(MID),"
			"UNIQUE(TID, Type),"
			"FOREIGN KEY(TID) REFERENCES Texture(TID) ON DELETE CASCADE, FOREIGN KEY(TGID) REFERENCES TextureGroup(TGID) ON DELETE CASCADE"
		");";

	//create table
	//if no error occurs, error message will be set to nullptr
	//normally there should be no exception
	this->Database->createFromSchema(TextureDatabaseSchema);
}

STPTextureDatabase::STPTextureDatabase(STPTextureDatabase&&) noexcept = default;

STPTextureDatabase& STPTextureDatabase::operator=(STPTextureDatabase&&) noexcept = default;

STPTextureDatabase::~STPTextureDatabase() = default;

STPTextureDatabase::STPTextureSplatBuilder& STPTextureDatabase::getSplatBuilder() {
	return const_cast<STPTextureSplatBuilder&>(const_cast<const STPTextureDatabase*>(this)->getSplatBuilder());
}

const STPTextureDatabase::STPTextureSplatBuilder& STPTextureDatabase::getSplatBuilder() const {
	return this->SplatBuilder;
}

STPTextureInformation::STPTextureGroupID STPTextureDatabase::addGroup(const STPTextureDescription& desc) {
	static constexpr string_view AddGroup =
		"INSERT INTO TextureGroup (TGID, Description) VALUES(?, ?);";
	sqlite3_stmt* const group_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddGroup, AddGroup);

	//insert texture desc as a binary
	STPsqliteCheckErr(sqlite3_bind_int(group_stmt, 1, static_cast<int>(++STPTextureDatabase::GroupIDAccumulator)));
	//we tell sqlite to copy the object and it will manage its lifetime for us
	STPsqliteCheckErr(sqlite3_bind_blob(group_stmt, 2, &desc, sizeof(desc), SQLITE_TRANSIENT));

	this->Database->execStmt(group_stmt);
	return STPTextureDatabase::GroupIDAccumulator;
}

void STPTextureDatabase::removeGroup(STPTextureInformation::STPTextureGroupID group_id) {
	static constexpr string_view RemoveGroup = 
		"DELETE FROM TextureGroup WHERE TGID = ?;";
	sqlite3_stmt* const group_stmt = this->Database->getStmt(STPTextureDatabaseImpl::RemoveGroup, RemoveGroup);

	//bind group id
	STPsqliteCheckErr(sqlite3_bind_int(group_stmt, 1, static_cast<int>(group_id)));

	this->Database->execStmt(group_stmt);
}

STPTextureDatabase::STPDatabaseView STPTextureDatabase::visit() const {
	return STPDatabaseView(*this);
}

STPTextureDatabase::STPTextureDescription STPTextureDatabase::getGroupDescription(STPTextureInformation::STPTextureGroupID id) const {
	static constexpr string_view GetGroup = 
		"SELECT Description FROM TextureGroup WHERE TGID = ?;";
	sqlite3_stmt* const group_stmt = this->Database->getStmt(STPTextureDatabaseImpl::GetGroup, GetGroup);

	STPsqliteCheckErr(sqlite3_bind_int(group_stmt, 1, static_cast<int>(id)));
	//now we need to retrieve the the blob data, since group ID is a primary key, we expect a unique result
	if (!this->Database->execStmt(group_stmt)) {
		//no data was retrieved, meaning group ID is invalid and nothing has been retrieved
		throw STPException::STPDatabaseError("Group ID given is undefined in the database");
	}

	//the lifetime of the blob pointer is temporary, we need to copy it
	return *reinterpret_cast<const STPTextureDatabase::STPTextureDescription*>(sqlite3_column_blob(group_stmt, 0));
}

size_t STPTextureDatabase::groupSize() const {
	static constexpr string_view GetGroupCount = 
		"SELECT COUNT(TGID) FROM TextureGroup;";

	return this->Database->getSingleOutput(GetGroupCount);
}

STPTextureInformation::STPTextureID STPTextureDatabase::addTexture() {
	STPTextureInformation::STPTextureID newID;
	this->addTexture(1u, &newID);
	return newID;
}

void STPTextureDatabase::addTexture(unsigned int count, STPTextureInformation::STPTextureID* texture_id) {
	if (count == 0u) {
		throw STPException::STPBadNumericRange("The number of texture ID requested must be a positive integer.");
	}

	static constexpr string_view AddTexture = 
		"INSERT INTO Texture (TID) VALUES(?);";
	sqlite3_stmt* texture_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddTexture, AddTexture);

	for (unsigned int i = 0u; i < count; i++) {
		//request a bunch of texture IDs
		texture_id[i] = STPTextureDatabase::TextureIDAccumulator++;
		STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 1, static_cast<int>(texture_id[i])));

		this->Database->execStmt(texture_stmt);
		//clear for the next iteration
		STPTextureDatabaseImpl::resetStmt(texture_stmt);
	}
}

void STPTextureDatabase::removeTexture(STPTextureInformation::STPTextureID texture_id) {
	static constexpr string_view RemoveTexture = 
		"DELETE FROM Texture WHERE TID = ?;";
	sqlite3_stmt* const texture_stmt = this->Database->getStmt(STPTextureDatabaseImpl::RemoveTexture, RemoveTexture);

	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 1, static_cast<int>(texture_id)));

	this->Database->execStmt(texture_stmt);
}

void STPTextureDatabase::addMap
	(STPTextureInformation::STPTextureID texture_id, STPTextureType type, STPTextureInformation::STPTextureGroupID group_id, const void* texture_data) {
	static constexpr string_view AddMap =
		"INSERT INTO Map (MID, Type, TGID, Data, TID) VALUES(?, ?, ?, ?, ?);";
	sqlite3_stmt* const texture_stmt = this->Database->getStmt(STPTextureDatabaseImpl::AddMap, AddMap);

	//set data
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 1, static_cast<int>(STPTextureDatabase::GeneralIDAccumulator++)));
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 2, static_cast<int>(type)));
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 3, static_cast<int>(group_id)));
	//send the pointer as a blob
	//here we have told user to manage the lifetime of texture for us, so sqlite doesn't need to worry about that
	//never assume the size of a pointer
	STPsqliteCheckErr(sqlite3_bind_blob(texture_stmt, 4, &texture_data, sizeof(texture_data), SQLITE_STATIC));
	STPsqliteCheckErr(sqlite3_bind_int(texture_stmt, 5, static_cast<int>(texture_id)));

	this->Database->execStmt(texture_stmt);
}

size_t STPTextureDatabase::mapSize() const {
	static constexpr string_view GetMapCount = 
		"SELECT COUNT(MID) FROM Map;";

	return this->Database->getSingleOutput(GetMapCount);
}

size_t STPTextureDatabase::textureSize() const {
	static constexpr string_view GetTextureCount = 
		"SELECT COUNT(TID) FROM Texture;";
	
	return this->Database->getSingleOutput(GetTextureCount);
}