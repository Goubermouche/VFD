#ifndef FILE_SYSTEM_H
#define FILE_SYSTEM_H

#include "pch.h"

#include "Core/Application.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <commdlg.h>
#include <GLFW/glfw3native.h>

namespace fe {
	class FileDialog {
	public:
		/// <summary>
		/// Opens a new file dialog and suspends the application, after the user selects a file the application resumes and a string containing the selected file path is returned. 
		/// </summary>
		/// <param name="filter">File extension filter.</param>
		/// <returns>String containing the selected item's file path, if no item was selected an empty string is returned.</returns>
		static std::string OpenFile(const char* filter) {
			OPENFILENAMEA ofn;
			CHAR szFile[260] = { 0 };

			ZeroMemory(&ofn, sizeof(OPENFILENAME));
			ofn.lStructSize = sizeof(OPENFILENAME);
			ofn.hwndOwner = glfwGetWin32Window((GLFWwindow*)Application::Get().GetWindow().GetNativeWindow());
			ofn.lpstrFile = szFile;
			ofn.nMaxFile = sizeof(szFile);
			ofn.lpstrFilter = filter;
			ofn.nFilterIndex = 1;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

			if (GetOpenFileNameA(&ofn) == TRUE) {
				return ofn.lpstrFile;
			}

			return std::string();
		}

		/// <summary>
		/// Opens a new file dialog and suspends the application, after the user selects a file the application resumes and a string containing the selected file path is returned. 
		/// </summary>
		/// <param name="filter">File extension filter.</param>
		/// <param name="defaultExtension">Default file extension.</param>
		/// <returns></returns>
		static std::string SaveFile(const char* filter, const char* defaultExtension = nullptr) {
			OPENFILENAMEA ofn;
			CHAR szFile[260] = { 0 };

			ZeroMemory(&ofn, sizeof(OPENFILENAME));
			ofn.lStructSize = sizeof(OPENFILENAME);
			ofn.hwndOwner = glfwGetWin32Window((GLFWwindow*)Application::Get().GetWindow().GetNativeWindow());
			ofn.lpstrFile = szFile;
			ofn.nMaxFile = sizeof(szFile);
			ofn.lpstrFilter = filter;
			ofn.nFilterIndex = 1;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
			ofn.lpstrDefExt = defaultExtension;

			if (GetSaveFileNameA(&ofn) == TRUE) {
				return ofn.lpstrFile;
			}

			return std::string();
		}
	};

	static bool FileExists(const std::string& filepath) {
		struct stat buffer;
		return (stat(filepath.c_str(), &buffer) == 0);
	}

	static std::string FilenameFromFilepath(const std::string& filepath) {
		return std::filesystem::path(filepath).stem().string();
	}

	static std::string FormatFileSize(uint64_t bytes) {
		float value = static_cast<double>(bytes);

		static const char* numberFormats[] = {
			"B", "K", "MB", "GB"
		};

		int index = 0;
		while ((value / 1024.0f) >= 1) {
			value = value / 1024.0f;
			index++;
		}

		return std::format("{} {}", std::ceil(value), numberFormats[index]);
	}
}

#endif // !FILE_SYSTEM_H