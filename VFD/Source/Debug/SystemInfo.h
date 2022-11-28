#ifndef SYSTEM_INFO_H
#define SYSTEM_INFO_H

namespace vfd {

	struct DeviceInfo {
		std::string Name;
		bool Valid = false;     // Viable CUDA device
		uint64_t ClockRate;     // KHz
		uint64_t GlobalMemory;  // Bytes
	};

	struct HostInfo {
		std::string Name;
		uint16_t CoreCount;    // After hyperthreading
		uint64_t GlobalMemory; // Bytes
	};

	struct OSInfo {
		std::string Name;
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