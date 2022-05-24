set(STP_RESOURCE_CACHE_LOCATION ${CMAKE_BINARY_DIR}/STPResourceCache)

# This helper function downloads resource from a given link and put it into the download cache directory.
# If a given file already exists in the download cache, function will silently skip it.
# Filename should not contain any extension which is specified separately.
function(downloadResource)
	set(Option EXTRACT)
	set(OneValueArg URL FILENAME EXTENSION)
	cmake_parse_arguments(Download_Resource "${Option}" "${OneValueArg}" "" ${ARGN})

	set(TargetFilenameRaw "${STP_RESOURCE_CACHE_LOCATION}/${Download_Resource_FILENAME}")
	set(TargetFilename "${TargetFilenameRaw}${Download_Resource_EXTENSION}")

	# check existence
	if(EXISTS ${TargetFilename} OR EXISTS ${TargetFilenameRaw})
		# exist? skip it
		message(STATUS "\'${TargetFilenameRaw}\' is up to date")
		return()
	endif()

	# non existing? download it
	message(STATUS "Downloading \'${Download_Resource_URL}\' to \'${TargetFilename}\'")
	file(
	DOWNLOAD ${Download_Resource_URL} ${TargetFilename}
	INACTIVITY_TIMEOUT 30
	STATUS DownloadStatus
	SHOW_PROGRESS
	)
	# error handling
	list(GET DownloadStatus 0 DownloadError)
	if(NOT ${DownloadError} EQUAL 0)
		list(GET DownloadStatus 1 ErrorLog)

		message(SEND_ERROR
		" Fail to download the file as requested\n"
		" Status code: ${DownloadError}\n"
		" Reason: ${ErrorLog}"
		)
		# remove the broken file
		file(REMOVE ${TargetFilename})

		return()
	endif()

	if(${Download_Resource_EXTRACT})
		message(STATUS "Extract \'${TargetFilename}\'")
		# extract as an archive
		file(ARCHIVE_EXTRACT
		INPUT ${TargetFilename}
		DESTINATION ${TargetFilenameRaw}
		)
		# clean up
		file(REMOVE ${TargetFilename})
	endif()
endfunction()