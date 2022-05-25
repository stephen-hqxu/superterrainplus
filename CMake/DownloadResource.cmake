set(STP_RESOURCE_CACHE_LOCATION ${CMAKE_BINARY_DIR}/STPResourceCache)

#[[
@brief This helper function downloads resource from a given link and put it into the download cache directory.
	If a given file already exists in the download cache, function will silently skip it.
------------------------------------------------------------------------------------------------
@param EXTRACT - If the downloaded file is an archive, extract the archive.
	Extracted file is put to a directory with the same name of the file, with extension excluded.
	The original archive file is deleted.
@param URL - provides a download link where the file should be downloaded.
@param FILENAME - should not contain any extension which is specified separately.
@param EXTENSION - is for the extension of downloaded file; extension is appended directly to the filename.
@param LOCATION - provides an output about the full directory location of downloaded file.
]]
function(downloadResource)
	set(Option EXTRACT)
	set(OneValueArg URL FILENAME EXTENSION LOCATION)
	cmake_parse_arguments(Download_Resource "${Option}" "${OneValueArg}" "" ${ARGN})

	# no extension
	set(TargetFilenameRaw "${STP_RESOURCE_CACHE_LOCATION}/${Download_Resource_FILENAME}")
	# contains extension
	set(TargetFilename "${TargetFilenameRaw}${Download_Resource_EXTENSION}")

	# write output location
	if(DEFINED Download_Resource_LOCATION)
		if(Download_Resource_EXTRACT)
			set(${Download_Resource_LOCATION} ${TargetFilenameRaw} PARENT_SCOPE)
		else()
			set(${Download_Resource_LOCATION} ${TargetFilename} PARENT_SCOPE)
		endif()
	endif()

	# check existence
	# extracted file is put into a directory with given filename without extension
	if((NOT Download_Resource_EXTRACT AND EXISTS ${TargetFilename}) OR (Download_Resource_EXTRACT AND EXISTS ${TargetFilenameRaw}))
		# exist? skip it
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
	if(NOT DownloadError EQUAL 0)
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

	if(Download_Resource_EXTRACT)
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