#ifndef FILE_SYSTEM_H_
#define FILE_SYSTEM_H_

#include <iostream>
#include <string>
#include <filesystem>

namespace fe {
	class FileDialog {
	public:
		/// <summary>
		/// Opens a new file dialog and suspends the application, after the user selects a file the application resumes and a string containing the selected file path is retuned. 
		/// </summary>
		/// <param name="filter">File extension filter</param>
		/// <returns>String containing the selected item's file path, if no item was selected an empty string is returned.</returns>
		static std::string OpenFile(const char* filter);

		/// <summary>
		/// Opens a new file dialog and suspends the application, after the user selects a file the application resumes and a string containing the selected file path is retuned. 
		/// </summary>
		/// <param name="filter">File extension filter.</param>
		/// <param name="defaultExtension">Default file extension.</param>
		/// <returns></returns>
		static std::string SaveFile(const char* filter, const char* defaultExtension = NULL);
	};

	static bool FileExists(const std::string& filepath) {
		struct stat buffer;
		return (stat(filepath.c_str(), &buffer) == 0);
	}

	static std::string FilenameFromFilepath(const std::string& filepath) {
		return std::filesystem::path(filepath).stem().string();
	}
}

#endif // !FILE_SYSTEM_H_