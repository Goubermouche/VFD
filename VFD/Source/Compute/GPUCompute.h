#ifndef COMPUTE_H_
#define COMPUTE_H_

namespace vfd {
	struct DeviceInfo {
		std::string Name;
		size_t ClockRate; // KHz
		size_t GlobalMemory; // Bytes
	};

	class GPUCompute
	{
	public:
		static void Init();
		static void Shutdown();

	    static DeviceInfo GetDeviceInfo() {
			return s_DeviceInfo;
		}

		static bool GetInitState() {
			return s_Initialized;
		}
	private:
		static DeviceInfo s_DeviceInfo;
		static bool s_Initialized;
	};
}

#endif // !COMPUTE_H_