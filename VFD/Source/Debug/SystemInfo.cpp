#include "pch.h"
#include "SystemInfo.h"

#include <cuda_runtime.h>
#include "Utility/FileSystem.h"

namespace vfd {
	DeviceInfo SystemInfo::s_DeviceInfo;
	HostInfo SystemInfo::s_HostInfo;
	OSInfo SystemInfo::s_OSInfo;

	void SystemInfo::Init()
	{
		InitDevice();
		InitHost();
		InitOS();
	}

	void SystemInfo::Shutdown()
	{
		if (HasValidCUDADevice()) {
			if (cudaDeviceReset() != cudaSuccess) {
				ASSERT("failed to shutdown CUDA!");
			}
		}
	}

	void SystemInfo::InitDevice()
	{
		int deviceCount;

		if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0); // Choose the first available device

			s_DeviceInfo.Name         = static_cast<std::string>(prop.name);
			s_DeviceInfo.ClockRate    = static_cast<uint32_t>(prop.clockRate);
			s_DeviceInfo.GlobalMemory = static_cast<uint64_t>(prop.totalGlobalMem);
			s_DeviceInfo.Valid        = true;

			std::cout << "Device: " << s_DeviceInfo.Name << '\n'
				      << "Device memory: " << fs::FormatFileSize(s_DeviceInfo.GlobalMemory) << '\n';

		}
		else {
			std::cout << "Device: no CUDA-capable device was found\n";
		}
	}

	void SystemInfo::InitHost()
	{
		int CPUInfo[4] = { -1 };
		unsigned int nExIds;
		char CPUBrandString[0x40];

		__cpuid(CPUInfo, 0x80000000);
		nExIds = CPUInfo[0];

		for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
			__cpuid(CPUInfo, i);
			switch (i)
			{
			case 0x80000002:
				memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
				break;
			case 0x80000003:
				memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
				break;
			case 0x80000004:
				memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
				break;
			}
		}

		MEMORYSTATUSEX statex;
		statex.dwLength = sizeof(statex);
		GlobalMemoryStatusEx(&statex);

		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);

		s_HostInfo.Name         = static_cast<std::string>(CPUBrandString);
		s_HostInfo.CoreCount    = static_cast<uint16_t>(sysInfo.dwNumberOfProcessors);
		s_HostInfo.GlobalMemory = static_cast<uint64_t>(statex.ullTotalPhys);

		std::cout << "Host: " << s_HostInfo.Name << '\n'
			      << "Host core count: " << s_HostInfo.CoreCount << '\n'
			      << "Host memory: " << fs::FormatFileSize(s_HostInfo.GlobalMemory) << '\n';
	}

	void SystemInfo::InitOS()
	{
#ifdef _WIN32
		s_OSInfo.Name = "Windows 32-bit";
#elif _WIN64
		s_OSInfo.Name = "Windows 64-bit";
#elif __APPLE__ || __MACH__
		s_OSInfo.Name = "Mac OSX";
#elif __linux__
		s_OSInfo.Name = "Linux";
#elif __FreeBSD__
		s_OSInfo.Name = "FreeBSD";
#elif __unix || __unix__
		s_OSInfo.Name = "Unix";
#else
		s_OSInfo.Name = "Other";
#endif

		std::cout << "OS: " << s_OSInfo.Name << '\n';
	}
}