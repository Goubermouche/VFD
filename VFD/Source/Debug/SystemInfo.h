#ifndef SYSTEM_INFO_H
#define SYSTEM_INFO_H

namespace vfd {
	struct DeviceInfo {
		std::string Name;
		bool Valid = false;                    // Viable CUDA device
		uint64_t ClockRate = 0u;               // KHz
		uint64_t GlobalMemory = 0u;            // Bytes
		uint32_t WarpSize = 0u;	               // Threads
		uint32_t MaxThreadsPerBlockCount = 0u; // Threads
		bool UnifiedAddressing = false;
		bool ManagedMemory = false;
		uint32_t VersionMajor = 0u;
		uint32_t VersionMinor = 0u;
	};

	struct HostInfo {
		std::string Name;
		uint16_t CoreCount = 0u;    // After hyperthreading
		uint64_t GlobalMemory = 0u; // Bytes
	};

	struct OSInfo {
		std::string Name;
		uint32_t VersionMajor = 0u;
		uint32_t VersionMinor = 0u;
		uint32_t Build = 0u;
	};

	class SystemInfo
	{
	public:
		static void Init();
		static void Shutdown();

		static const DeviceInfo& GetDeviceInfo() {
			return s_DeviceInfo;
		}

		static const HostInfo& GetHostInfo() {
			return s_HostInfo;
		}

		static const bool HasValidCUDADevice() {
			return s_DeviceInfo.Valid;
		}

		static const bool CUDADeviceMeetsRequirements() {
			return s_DeviceInfo.Valid /*&& s_DeviceInfo.UnifiedAddressing && s_DeviceInfo.ManagedMemory*/;
		}
	private:
		static void InitDevice();
		static void InitHost();
		static void InitOS();

		static DeviceInfo s_DeviceInfo;
		static HostInfo s_HostInfo;
		static OSInfo s_OSInfo;
	};
}

#endif // !SYSTEM_INFO_H