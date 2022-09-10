#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/Grid3D.cuh"
#include "Simulation/FLIP/Utility/Interpolation.cuh"

namespace fe {
	struct ValidVelocityComponentGrid {
		__host__ __device__ void Init(int i, int j, int k) {
			ValidU.Init(i + 1, j, k, false);
			ValidV.Init(i, j + 1, k, false);
			ValidW.Init(i, j, k + 1, false);
		}

		__host__ __device__ void Reset() {
			ValidU.Fill(false);
			ValidV.Fill(false);
			ValidW.Fill(false);
		}

		__host__ __device__ void HostFree() {
			ValidU.HostFree();
			ValidV.HostFree();
			ValidW.HostFree();
		}

		__host__ __device__ void DeviceFree() {
			ValidU.DeviceFree();
			ValidV.DeviceFree();
			ValidW.DeviceFree();
		}

		Array3D<bool> ValidU;
		Array3D<bool> ValidV;
		Array3D<bool> ValidW;
	};

	enum MACGridStatus : char {
		Unknown = 0x00,
		Waiting = 0x01,
		Known = 0x02,
		Done = 0x03
	};

	struct MACVelocityField {
		__host__ __device__ void Init(int i, int j, int k, double dx) {
			Size = { i, j, k };
			DX = dx;

			U.Init(Size.x + 1, Size.y, Size.z, 0.0f);
			V.Init(Size.x, Size.y + 1, Size.z, 0.0f);
			W.Init(Size.x, Size.y, Size.z + 1, 0.0f);

			U.SetOutOfRangeValue(0.0f);
			V.SetOutOfRangeValue(0.0f);
			W.SetOutOfRangeValue(0.0f);
		}

		__host__ __device__ bool IsIndexInRangeU(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x + 1, Size.y, Size.z);
		}

		__host__ __device__ bool IsIndexInRangeV(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x, Size.y + 1, Size.z);
		}

		__host__ __device__ bool IsIndexInRangeW(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x, Size.y, Size.z + 1);
		}

		__host__ __device__ void SetU(int i, int j, int k, double Value) {
			if (!IsIndexInRangeU(i, j, k)) {
				return;
			}

			U.Set(i, j, k, (float)Value);
		}

		__host__ __device__ void SetV(int i, int j, int k, double Value) {
			if (!IsIndexInRangeV(i, j, k)) {
				return;
			}

			V.Set(i, j, k, (float)Value);
		}

		__host__ __device__ void SetW(int i, int j, int k, double Value) {
			if (!IsIndexInRangeW(i, j, k)) {
				return;
			}

			W.Set(i, j, k, (float)Value);
		}

		__host__ __device__ void AddU(int i, int j, int k, double Value) {
			if (!IsIndexInRangeU(i, j, k)) {
				return;
			}

			U.Add(i, j, k, (float)Value);
		}

		__host__ __device__ void AddV(int i, int j, int k, double Value) {
			if (!IsIndexInRangeV(i, j, k)) {
				return;
			}

			V.Add(i, j, k, (float)Value);
		}

		__host__ __device__ void AddW(int i, int j, int k, double Value) {
			if (!IsIndexInRangeW(i, j, k)) {
				return;
			}

			W.Add(i, j, k, (float)Value);
		}

		__host__ __device__ void ClearU() {
			U.Fill(0.0f);
		}

		__host__ __device__ void ClearV() {
			V.Fill(0.0f);
		}

		__host__ __device__ void ClearW() {
			W.Fill(0.0f);
		}

		__host__ __device__ void Clear() {
			ClearU();
			ClearV();
			ClearW();
		}

		__host__ __device__ glm::vec3 EvaluateVelocityAtPositionLinear(glm::vec3 pos) {
			return EvaluateVelocityAtPositionLinear(pos.x, pos.y, pos.z);
		}

		__host__ __device__ glm::vec3 EvaluateVelocityAtPositionLinear(double x, double y, double z) {
			if (!IsPositionInGrid(x, y, z, DX, Size.x, Size.y, Size.z)) {
				return glm::vec3();
			}

			double xvel = InterpolateLinearU(x, y, z);
			double yvel = InterpolateLinearV(x, y, z);
			double zvel = InterpolateLinearW(x, y, z);

			return glm::vec3((float)xvel, (float)yvel, (float)zvel);
		}

		__host__ __device__ double InterpolateLinearU(double x, double y, double z) {
			if (!IsPositionInGrid(x, y, z, DX, Size.x, Size.y, Size.z)) {
				return 0.0;
			}

			y -= 0.5 * DX;
			z -= 0.5 * DX;

			int i;
			int j;
			int k;

			double gx;
			double gy;
			double gz;

			PositionToGridIndex(x, y, z, DX, &i, &j, &k);
			GridIndexToPosition(i, j, k, DX, &gx, &gy, &gz);

			double invDX = 1.0 / DX;
			double ix = (x - gx) * invDX;
			double iy = (y - gy) * invDX;
			double iz = (z - gz) * invDX;

			double points[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			if (U.IsIndexInRange(i,     j,     k    )) { points[0] = U(i,     j,     k    ); }
			if (U.IsIndexInRange(i + 1, j,     k    )) { points[1] = U(i + 1, j,     k    ); }
			if (U.IsIndexInRange(i,     j + 1, k    )) { points[2] = U(i,     j + 1, k    ); }
			if (U.IsIndexInRange(i,     j,     k + 1)) { points[3] = U(i,     j,     k + 1); }
			if (U.IsIndexInRange(i + 1, j,     k + 1)) { points[4] = U(i + 1, j,     k + 1); }
			if (U.IsIndexInRange(i,     j + 1, k + 1)) { points[5] = U(i,     j + 1, k + 1); }
			if (U.IsIndexInRange(i + 1, j + 1, k    )) { points[6] = U(i + 1, j + 1, k    ); }
			if (U.IsIndexInRange(i + 1, j + 1, k + 1)) { points[7] = U(i + 1, j + 1, k + 1); }

			return Interpolation::TrilinearInterpolate(points, ix, iy, iz);
		}

		__host__ __device__ double InterpolateLinearV(double x, double y, double z) {
			if (!IsPositionInGrid(x, y, z, DX, Size.x, Size.y, Size.z)) {
				return 0.0;
			}

			x -= 0.5 * DX;
			z -= 0.5 * DX;

			int i;
			int j;
			int k;

			double gx;
			double gy;
			double gz;

			PositionToGridIndex(x, y, z, DX, &i, &j, &k);
			GridIndexToPosition(i, j, k, DX, &gx, &gy, &gz);

			double invDX = 1 / DX;
			double ix = (x - gx) * invDX;
			double iy = (y - gy) * invDX;
			double iz = (z - gz) * invDX;

			double points[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			if (V.IsIndexInRange(i,     j,     k    )) { points[0] = V(i,     j,     k    ); }
			if (V.IsIndexInRange(i + 1, j,     k    )) { points[1] = V(i + 1, j,     k    ); }
			if (V.IsIndexInRange(i,     j + 1, k    )) { points[2] = V(i,     j + 1, k    ); }
			if (V.IsIndexInRange(i,     j,     k + 1)) { points[3] = V(i,     j,     k + 1); }
			if (V.IsIndexInRange(i + 1, j,     k + 1)) { points[4] = V(i + 1, j,     k + 1); }
			if (V.IsIndexInRange(i,     j + 1, k + 1)) { points[5] = V(i,     j + 1, k + 1); }
			if (V.IsIndexInRange(i + 1, j + 1, k    )) { points[6] = V(i + 1, j + 1, k    ); }
			if (V.IsIndexInRange(i + 1, j + 1, k + 1)) { points[7] = V(i + 1, j + 1, k + 1); }

			return Interpolation::TrilinearInterpolate(points, ix, iy, iz);
		}

		__host__ __device__ double InterpolateLinearW(double x, double y, double z) {
			if (!IsPositionInGrid(x, y, z, DX, Size.x, Size.y, Size.z)) {
				return 0.0;
			}

			x -= 0.5 * DX;
			y -= 0.5 * DX;

			int i;
			int j; 
			int k;

			double gx;
			double gy;
			double gz;

			PositionToGridIndex(x, y, z, DX, &i, &j, &k);
			GridIndexToPosition(i, j, k, DX, &gx, &gy, &gz);

			double invDX = 1 / DX;
			double ix = (x - gx) * invDX;
			double iy = (y - gy) * invDX;
			double iz = (z - gz) * invDX;

			double points[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			if (W.IsIndexInRange(i,     j,     k    )) { points[0] = W(i,     j,     k    ); }
			if (W.IsIndexInRange(i + 1, j,     k    )) { points[1] = W(i + 1, j,     k    ); }
			if (W.IsIndexInRange(i,     j + 1, k    )) { points[2] = W(i,     j + 1, k    ); }
			if (W.IsIndexInRange(i,     j,     k + 1)) { points[3] = W(i,     j,     k + 1); }
			if (W.IsIndexInRange(i + 1, j,     k + 1)) { points[4] = W(i + 1, j,     k + 1); }
			if (W.IsIndexInRange(i,     j + 1, k + 1)) { points[5] = W(i,     j + 1, k + 1); }
			if (W.IsIndexInRange(i + 1, j + 1, k    )) { points[6] = W(i + 1, j + 1, k    ); }
			if (W.IsIndexInRange(i + 1, j + 1, k + 1)) { points[7] = W(i + 1, j + 1, k + 1); }

			return Interpolation::TrilinearInterpolate(points, ix, iy, iz);
		}

		__host__ __device__ void ExtrapolateGrid(Array3D<float>& grid, Array3D<bool>& valid, int numLayers) {
			Array3D<MACGridStatus> status;
			status.Init(grid.Size.x, grid.Size.y, grid.Size.z);

			for (int k = 0; k < grid.Size.z; k++) {
				for (int j = 0; j < grid.Size.y; j++) {
					for (int i = 0; i < grid.Size.x; i++) {
						status.Set(i, j, k, valid(i, j, k) ? MACGridStatus::Known : MACGridStatus::Unknown);
						if (status(i, j, k) == MACGridStatus::Unknown && IsGridIndexOnBorder(i, j, k, grid.Size.x, grid.Size.y, grid.Size.z)) {
							status.Set(i, j, k, MACGridStatus::Done);
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
							if (status(i, j, k) != MACGridStatus::Known) {
								continue;
							}

							int count = 0;
							if (status(i - 1, j, k) == MACGridStatus::Unknown) {
								extrapolationCells.push_back(glm::ivec3(i - 1, j, k));
								status.Set(i - 1, j, k, MACGridStatus::Waiting);
								count++;
							}
							else if (status(i - 1, j, k) == MACGridStatus::Waiting) {
								count++;
							}

							if (status(i + 1, j, k) == MACGridStatus::Unknown) {
								extrapolationCells.push_back(glm::ivec3(i + 1, j, k));
								status.Set(i + 1, j, k, MACGridStatus::Waiting);
								count++;
							}
							else if (status(i + 1, j, k) == MACGridStatus::Waiting) {
								count++;
							}

							if (status(i, j - 1, k) == MACGridStatus::Unknown) {
								extrapolationCells.push_back(glm::ivec3(i, j - 1, k));
								status.Set(i, j - 1, k, MACGridStatus::Waiting);
								count++;
							}
							else if (status(i, j - 1, k) == MACGridStatus::Waiting) {
								count++;
							}

							if (status(i, j + 1, k) == MACGridStatus::Unknown) {
								extrapolationCells.push_back(glm::ivec3(i, j + 1, k));
								status.Set(i, j + 1, k, MACGridStatus::Waiting);
								count++;
							}
							else if (status(i, j + 1, k) == MACGridStatus::Waiting) {
								count++;
							}

							if (status(i, j, k - 1) == MACGridStatus::Unknown) {
								extrapolationCells.push_back(glm::ivec3(i, j, k - 1));
								status.Set(i, j, k - 1, MACGridStatus::Waiting);
								count++;
							}
							else if (status(i, j, k - 1) == MACGridStatus::Waiting) {
								count++;
							}

							if (status(i, j, k + 1) == MACGridStatus::Unknown) {
								extrapolationCells.push_back(glm::ivec3(i, j, k + 1));
								status.Set(i, j, k + 1, MACGridStatus::Waiting);
								count++;
							}
							else if (status(i, j, k + 1) == MACGridStatus::Waiting) {
								count++;
							}

							if (count == 0) {
								status.Set(i, j, k, MACGridStatus::Done);
							}
						}
					}
				}

				glm::ivec3 g;
				for (size_t i = 0; i < extrapolationCells.size(); i++) {
					g = extrapolationCells[i];
					float sum = 0;
					int count = 0;

					if (status(g.x - 1, g.y,     g.z    ) == MACGridStatus::Known) { sum += grid(g.x - 1, g.y,     g.z    ); count++; }
					if (status(g.x + 1, g.y,     g.z    ) == MACGridStatus::Known) { sum += grid(g.x + 1, g.y,     g.z    ); count++; }
					if (status(g.x,     g.y - 1, g.z    ) == MACGridStatus::Known) { sum += grid(g.x,     g.y - 1, g.z    ); count++; }
					if (status(g.x,     g.y + 1, g.z    ) == MACGridStatus::Known) { sum += grid(g.x,     g.y + 1, g.z    ); count++; }
					if (status(g.x,     g.y,     g.z - 1) == MACGridStatus::Known) { sum += grid(g.x,     g.y,     g.z - 1); count++; }
					if (status(g.x,     g.y,     g.z + 1) == MACGridStatus::Known) { sum += grid(g.x,     g.y,     g.z + 1); count++; }

					grid.Set(g, sum / (float)count);
				}

				status.Set(extrapolationCells, MACGridStatus::Known);
			}

			status.HostFree();
		}

		__host__ __device__ void ExtrapolateVelocityField(ValidVelocityComponentGrid& validGrid, int numLayers) {
			ExtrapolateGrid(U, validGrid.ValidU, numLayers);
			ExtrapolateGrid(V, validGrid.ValidV, numLayers);
			ExtrapolateGrid(W, validGrid.ValidW, numLayers);
		}

		__host__ __device__ void DeviceFree() {
			U.DeviceFree();
			V.DeviceFree();
			W.DeviceFree();
		}

		__host__ __device__ void HostFree() {
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