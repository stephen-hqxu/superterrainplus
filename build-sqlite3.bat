REM * This is a helper script for building a minimal and fastest SQLite3 database library for SuperTerrain+ on Windows,
REM		by omitting certain features the engine does not require.
REM * Please run the script at the as directory as amalgamated SQLite3 source files.
REM * It is not mandatory to build the library with this script, it is compatible with default-compiled settings.
REM * The script needs to be executed from Native Tools Command Prompt for Visual Studio, either x86 or x64.
@echo OFF

REM Please note that the engine by default uses SQLite3 on single-threaded environment therefore multithreading support has been omitted.
REM If you with to develop your application requires multithreaded database please enable thread safe build option.
REM Likewise, please refer to SQLite3 documentation to customise other compile-time options as needed.
set compile_time_option=/DSQLITE_API=__declspec(dllexport) /DSQLITE_OMIT_DEPRECATED /DSQLITE_OMIT_AUTOINIT /DSQLITE_USE_ALLOCA /DSQLITE_DQS=0 /DSQLITE_THREADSAFE=0 /DSQLITE_OMIT_SHARED_CACHE /DSQLITE_LIKE_DOESNT_MATCH_BLOBS
set compiler_flag=/O2 /Ox /MD

cl %compile_time_option% %compiler_flag% sqlite3.c -link -dll -out:sqlite3.dll