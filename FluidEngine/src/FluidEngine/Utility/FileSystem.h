#ifndef FILE_SYSTEM_H_
#define FILE_SYSTEM_H_

namespace fe {
	class FileDialog {
		static std::string OpenFile(const char* filter);
		static std::string SaveFile(const char* filter);
	};
}

#endif // !FILE_SYSTEM_H_
