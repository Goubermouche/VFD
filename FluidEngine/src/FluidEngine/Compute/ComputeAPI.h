#ifndef COMPUTE_API_H
#define COMPUTE_API_H

namespace fe {
	enum class ComputeAPIType {
		None, 
		CUDA
		// OpenCL
	};

	struct DeviceInfo {
		std::string name;
		int clockRate; // hz
		int globalMemory; // bytes
		bool concurrentKernels;
		int coreCount;
	};

	/// <summary>
	/// Global compute API. Serves as an intermediate between the Compute class and the various platform compute classes.
	/// </summary>
	class ComputeAPI
	{
	public:
		virtual void Init() = 0;

		static inline ComputeAPIType GetAPIType() {
			return s_API;
		}

		static void SetAPI(ComputeAPIType api);

		static bool GetInitState() {
			return s_InitializedSuccessfully;
		}

		static DeviceInfo GetDeviceInfo() {
			return s_DeviceInfo;
		}
	protected:
		static ComputeAPIType s_API;
		static DeviceInfo s_DeviceInfo;
		static bool s_InitializedSuccessfully;
	};
}

#endif // !COMPUTE_API_H