#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/Grid3D.cuh"
#include "Simulation/FLIP/Utility/Interpolation.cuh"

namespace fe {
	struct ValidVelocityComponentGrid {
		__device__ ValidVelocityComponentGrid() {}

		__device__ void Init(int i, int j, int k) {
			ValidU.Init(i + 1, j, k, false);
			ValidV.Init(i, j + 1, k, false);
			ValidW.Init(i, j, k + 1, false);
		}

		__device__ void Reset() {
			ValidU.Fill(false);
			ValidV.Fill(false);
			ValidW.Fill(false);
		}

		// TODO: add free methods

		Array3D<bool> ValidU;
		Array3D<bool> ValidV;
		Array3D<bool> ValidW;
	};

	struct MACVelocityField {
		__device__ MACVelocityField() {}

		__device__ void Init(int i, int j, int k, double dx) {
			Size = { i, j, k };
			DX = dx;

			U.Init(Size.x + 1, Size.y, Size.z, 0.0f);
			V.Init(Size.x, Size.y + 1, Size.z, 0.0f);
			W.Init(Size.x, Size.y, Size.z + 1, 0.0f);

			U.SetOutOfRangeValue(0.0f);
			V.SetOutOfRangeValue(0.0f);
			W.SetOutOfRangeValue(0.0f);

			LOG("velocity grids initialized", "FLIP", ConsoleColor::Cyan);
		}

		__host__ MACVelocityField UploadToDevice() {
			MACVelocityField device = *this;

			// Perform a deep copy on the individual arrays
			device.U.UploadToDevice(device.U);
			device.V.UploadToDevice(device.V);
			device.W.UploadToDevice(device.W);

			return device;
		}

		__device__ __host__ bool IsIndexInRangeU(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x + 1, Size.y, Size.z);
		}
		__device__ __host__ bool IsIndexInRangeV(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x, Size.y + 1, Size.z);
		}
		__device__ __host__ bool IsIndexInRangeW(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x, Size.y, Size.z + 1);
		}

		__device__ __host__ void SetU(int i, int j, int k, double val) {
			if (!IsIndexInRangeU(i, j, k)) {
				return;
			}

			U.Set(i, j, k, (float)val);
		}

		__device__ __host__ void SetV(int i, int j, int k, double val) {
			if (!IsIndexInRangeV(i, j, k)) {
				return;
			}

			V.Set(i, j, k, (float)val);
		}

		__device__ __host__ void SetW(int i, int j, int k, double val) {
			if (!IsIndexInRangeW(i, j, k)) {
				return;
			}

			W.Set(i, j, k, (float)val);
		}

		__device__ __host__ void AddU(int i, int j, int k, double val) {
			if (!IsIndexInRangeU(i, j, k)) {
				return;
			}

			U.Add(i, j, k, (float)val);
		}

		__device__ __host__ void AddV(int i, int j, int k, double val) {
			if (!IsIndexInRangeV(i, j, k)) {
				return;
			}

			V.Add(i, j, k, (float)val);
		}

		__device__ __host__ void AddW(int i, int j, int k, double val) {
			if (!IsIndexInRangeW(i, j, k)) {
				return;
			}

			W.Add(i, j, k, (float)val);
		}

		__device__ __host__ void ClearU() {
			U.Fill(0.0);
		}

		__device__ __host__ void ClearV() {
			V.Fill(0.0);
		}

		__device__ __host__ void ClearW() {
			W.Fill(0.0);
		}

		__device__ __host__ void Clear() {
			ClearU();
			ClearV();
			ClearW();
		}

		__device__ __host__ void ExtrapolateGrid(Array3D<float>& grid, Array3D<bool>& valid, int numLayers) {
			char UNKNOWN = 0x00;
			char WAITING = 0x01;
			char KNOWN = 0x02;
			char DONE = 0x03;

			Array3D<char> status;
			status.Init(grid.Size.x, grid.Size.y, grid.Size.z);

			for (int k = 0; k < grid.Size.z; k++) {
				for (int j = 0; j < grid.Size.y; j++) {
					for (int i = 0; i < grid.Size.x; i++) {
						status.Set(i, j, k, valid(i, j, k) ? KNOWN : UNKNOWN);
						if (status(i, j, k) == UNKNOWN &&
							IsGridIndexOnBorder(i, j, k, grid.Size.x, grid.Size.y, grid.Size.z)) {
							status.Set(i, j, k, DONE);
						}
					}
				}
			}

			std::vector<glm::ivec3> extrapolationCells;
			for (int layers = 0; layers < numLayers; layers++) {

				extrapolationCells.clear();
				for (int k = 1; k < grid.Size.z - 1; k++) {
					for (int j = 1; j < grid.Size.y - 1; j++) {
						for (int i = 1; i < grid.Size.x - 1; i++) {
							if (status(i, j, k) != KNOWN) {
								continue;
							}

							int count = 0;
							if (status(i - 1, j, k) == UNKNOWN) {
								extrapolationCells.push_back(glm::ivec3(i - 1, j, k));
								status.Set(i - 1, j, k, WAITING);
								count++;
							}
							else if (status(i - 1, j, k) == WAITING) {
								count++;
							}

							if (status(i + 1, j, k) == UNKNOWN) {
								extrapolationCells.push_back(glm::ivec3(i + 1, j, k));
								status.Set(i + 1, j, k, WAITING);
								count++;
							}
							else if (status(i + 1, j, k) == WAITING) {
								count++;
							}

							if (status(i, j - 1, k) == UNKNOWN) {
								extrapolationCells.push_back(glm::ivec3(i, j - 1, k));
								status.Set(i, j - 1, k, WAITING);
								count++;
							}
							else if (status(i, j - 1, k) == WAITING) {
								count++;
							}

							if (status(i, j + 1, k) == UNKNOWN) {
								extrapolationCells.push_back(glm::ivec3(i, j + 1, k));
								status.Set(i, j + 1, k, WAITING);
								count++;
							}
							else if (status(i, j + 1, k) == WAITING) {
								count++;
							}

							if (status(i, j, k - 1) == UNKNOWN) {
								extrapolationCells.push_back(glm::ivec3(i, j, k - 1));
								status.Set(i, j, k - 1, WAITING);
								count++;
							}
							else if (status(i, j, k - 1) == WAITING) {
								count++;
							}

							if (status(i, j, k + 1) == UNKNOWN) {
								extrapolationCells.push_back(glm::ivec3(i, j, k + 1));
								status.Set(i, j, k + 1, WAITING);
								count++;
							}
							else if (status(i, j, k + 1) == WAITING) {
								count++;
							}

							if (count == 0) {
								status.Set(i, j, k, DONE);
							}
						}
					}
				}

				glm::ivec3 g;
				for (size_t i = 0; i < extrapolationCells.size(); i++) {
					g = extrapolationCells[i];

					float sum = 0;
					int count = 0;
					if (status(g.x - 1, g.y, g.z) == KNOWN) { sum += grid(g.x - 1, g.y, g.z); count++; }
					if (status(g.x + 1, g.y, g.z) == KNOWN) { sum += grid(g.x + 1, g.y, g.z); count++; }
					if (status(g.x, g.y - 1, g.z) == KNOWN) { sum += grid(g.x, g.y - 1, g.z); count++; }
					if (status(g.x, g.y + 1, g.z) == KNOWN) { sum += grid(g.x, g.y + 1, g.z); count++; }
					if (status(g.x, g.y, g.z - 1) == KNOWN) { sum += grid(g.x, g.y, g.z - 1); count++; }
					if (status(g.x, g.y, g.z + 1) == KNOWN) { sum += grid(g.x, g.y, g.z + 1); count++; }

					grid.Set(g, sum / (float)count);
				}

				status.Set(extrapolationCells, KNOWN);
			}

			status.HostFree();
		}

		__device__ __host__ void ExtrapolateVelocityField(ValidVelocityComponentGrid& validGrid, int numLayers) {
			ExtrapolateGrid(U, validGrid.ValidU, numLayers);
			ExtrapolateGrid(V, validGrid.ValidV, numLayers);
			ExtrapolateGrid(W, validGrid.ValidW, numLayers);
		}

		__host__ void DeviceFree() {
			U.DeviceFree();
			V.DeviceFree();
			W.DeviceFree();
		}

		__host__ void HostFree() {
			U.HostFree();
			V.HostFree();
			W.HostFree();
		}

		Array3D<float> U;
		Array3D<float> V;
		Array3D<float> W;

		float DefaultOutOfRangeValue;
		double DX;

		int ExtrapolationLayerCount;

		glm::ivec3 Size;
	};
}

#endif // !MARKER_AND_CELL_VELOCITY_FIELD_CUH