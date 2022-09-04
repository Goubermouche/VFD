#ifndef VISCOSITY_SOLVER_CUH
#define VISCOSITY_SOLVER_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/LevelSetUtils.cuh"
#include "Simulation/FLIP/Utility/SparseMatrix.cuh"
#include "Simulation/FLIP/Utility/PCGSolver.cuh"

namespace fe {
	struct ViscositySolverDescription {
		float cellWidth;
		float deltaTime;

		MACVelocityField* velocityField;
		ParticleLevelSet* liquidSDF;
		MeshLevelSet* solidSDF;
		Array3D<float>* viscosity;
	};

	enum class FaceState : char {
		AIR = 0x00,
		FLUID = 0x01,
		SOLID = 0x02
	};

	struct FaceStateGrid {
		glm::ivec3 Size;
		Array3D<FaceState> U;
		Array3D<FaceState> V;
		Array3D<FaceState> W;

		__device__ __host__ FaceStateGrid() {}

		__device__ __host__ void Init(int i, int j, int k) {
			Size = { i, j, k };
			U.Init(i + 1, j, k, FaceState::AIR);
			V.Init(i, j + 1, k, FaceState::AIR);
			W.Init(i, j, k + 1, FaceState::AIR);
		}

		__device__ __host__ void HostFree() {
			U.HostFree();
			V.HostFree();
			W.HostFree();
		}
	};

	struct ViscosityVolumeGrid {
		glm::ivec3 Size;
		Array3D<float> center;
		Array3D<float> U;
		Array3D<float> V;
		Array3D<float> W;
		Array3D<float> edgeU;
		Array3D<float> edgeV;
		Array3D<float> edgeW;

		__device__ __host__ ViscosityVolumeGrid() {}

		__device__ __host__ void Init(int i, int j, int k) {
			Size = { i, j, k };
			center.Init(i, j, k, 0.0f);
			U.Init(i + 1, j, k, 0.0f);
			V.Init(i, j + 1, k, 0.0f);
			W.Init(i, j, k + 1, 0.0f);
			edgeU.Init(i, j + 1, k + 1, 0.0f);
			edgeV.Init(i + 1, j, k + 1, 0.0f);
			edgeW.Init(i + 1, j + 1, k, 0.0f);
		}

		__device__ __host__ void HostFree() {
			center.HostFree();
			U.HostFree();
			V.HostFree();
			W.HostFree();
			edgeU.HostFree();
			edgeV.HostFree();
			edgeW.HostFree();
		}
	};

	struct FaceIndexer {
		glm::vec3 Size;

		__device__ __host__  FaceIndexer() {}
		__device__ __host__  void Init(int i, int j, int k) {
			Size = { i, j, k };
			_voffset = (Size.x + 1) * Size.y * Size.z;
			_woffset = _voffset + Size.x * (Size.y + 1) * Size.z;
		}

		__device__ __host__  int U(int i, int j, int k) {
			return i + (Size.x + 1) * (j + k * Size.y);
		}

		__device__ __host__  int V(int i, int j, int k) {
			return _voffset + i + Size.x * (j + k * (Size.y + 1));
		}

		__device__ __host__  int W(int i, int j, int k) {
			return _woffset + i + Size.x * (j + k * Size.z);
		}

	private:
		int _voffset;
		int _woffset;
	};

	struct MatrixIndexer {
		std::vector<int> indexTable;
		FaceIndexer faceIndexer;
		int matrixSize;

		__device__ __host__ MatrixIndexer() {}
		__device__ __host__ void Init(int i, int j, int k, std::vector<int> matrixIndexTable) {

			indexTable = matrixIndexTable;
			faceIndexer.Init(i, j, k);
			int matsize = 0;
			for (size_t i = 0; i < indexTable.size(); i++) {
				if (indexTable[i] != -1) {
					matsize++;
				}
			}

			matrixSize = matsize;
		}

		__device__ __host__ int U(int i, int j, int k) {
			return indexTable[faceIndexer.U(i, j, k)];
		}

		__device__ __host__ int V(int i, int j, int k) {
			return indexTable[faceIndexer.V(i, j, k)];
		}

		__device__ __host__ int W(int i, int j, int k) {
			return indexTable[faceIndexer.W(i, j, k)];
		}
	};

	struct ViscositySolver {
		__device__ __host__ bool ApplyViscosityToVelocityField(ViscositySolverDescription desc) {
			Init(desc);
			CalculateFaceStateGrid();
			CalculateVolumeGrid();
			CalculateMatrixIndexTable();

			int matsize = MatrixIndex.matrixSize;
			SparseMatrix<float> matrix(matsize);
			std::vector<float> rhs(matsize, 0);
			std::vector<float> soln(matsize, 0);

			InitializeLinearSystem(matrix, rhs);

			bool success = SolveLinearSystem(matrix, rhs, soln);
			if (success == false) {
				return false;
			}

			ApplySolutionToVelocityField(soln);

			return true;
		}

		__device__ __host__ void ApplySolutionToVelocityField(std::vector<float>& soln) {
			VelocityField->Clear();
			for (int k = 0; k < Size.z; k++) {
				for (int j = 0; j < Size.y; j++) {
					for (int i = 0; i < Size.x + 1; i++) {
						int matidx = MatrixIndex.U(i, j, k);
						if (matidx != -1) {
							VelocityField->SetU(i, j, k, soln[matidx]);
						}
					}
				}
			}

			for (int k = 0; k < Size.z; k++) {
				for (int j = 0; j < Size.y + 1; j++) {
					for (int i = 0; i < Size.x; i++) {
						int matidx = MatrixIndex.V(i, j, k);
						if (matidx != -1) {
							VelocityField->SetV(i, j, k, soln[matidx]);
						}
					}
				}
			}

			for (int k = 0; k < Size.z + 1; k++) {
				for (int j = 0; j < Size.y; j++) {
					for (int i = 0; i < Size.x; i++) {
						int matidx = MatrixIndex.W(i, j, k);
						if (matidx != -1) {
							VelocityField->SetW(i, j, k, soln[matidx]);
						}
					}
				}
			}
		}

		__device__ __host__ bool SolveLinearSystem(SparseMatrix<float>& matrix, std::vector<float>& rhs, std::vector<float>& soln) {
			PCGSolver<float> solver;
			solver.setSolverParameters(SolverTolerance, MaxSolverIterations);

			float estimatedError;
			int numIterations;
			bool success = solver.solve(matrix, rhs, soln, estimatedError, numIterations);

			if (success) {
				std::cout << "\n\tViscosity Solver Iterations: " << numIterations <<
					"\n\tEstimated Error: " << estimatedError << "\n\n";
				return true;
			}
			else if (numIterations == MaxSolverIterations && estimatedError < AcceptableTolerace) {
				std::cout << "\n\tViscosity Solver Iterations: " << numIterations <<
					"\n\tEstimated Error: " << estimatedError << "\n\n";
				return true;
			}
			else {
				std::cout << "\n\t***Viscosity Solver FAILED" <<
					"\n\tViscosity Solver Iterations: " << numIterations <<
					"\n\tEstimated Error: " << estimatedError << "\n\n";
				return false;
			}
		}

		__device__ __host__ void InitializeLinearSystem(SparseMatrix<float>& matrix, std::vector<float>& rhs) {
			InitializeLinearSystemU(matrix, rhs);
			InitializeLinearSystemV(matrix, rhs);
			InitializeLinearSystemW(matrix, rhs);
		}

		__device__ __host__ void InitializeLinearSystemU(SparseMatrix<float>& matrix, std::vector<float>& rhs) {
			MatrixIndexer& mj = MatrixIndex;
			FaceState FLUID = FaceState::FLUID;
			FaceState SOLID = FaceState::SOLID;

			float invdx = 1.0f / DX;
			float factor = DeltaTime * invdx * invdx;
			for (int k = 1; k < Size.z; k++) {
				for (int j = 1; j < Size.y; j++) {
					for (int i = 1; i < Size.x; i++) {

						if (State.U(i, j, k) != FaceState::FLUID) {
							continue;
						}

						int row = MatrixIndex.U(i, j, k);
						if (row == -1) {
							continue;
						}

						float viscRight = Viscosity->Get(i, j, k);
						float viscLeft = Viscosity->Get(i - 1, j, k);

						float viscTop = 0.25f * (Viscosity->Get(i - 1, j + 1, k) +
							Viscosity->Get(i - 1, j, k) +
							Viscosity->Get(i, j + 1, k) +
							Viscosity->Get(i, j, k));
						float viscBottom = 0.25f * (Viscosity->Get(i - 1, j, k) +
							Viscosity->Get(i - 1, j - 1, k) +
							Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j - 1, k));

						float viscFront = 0.25f * (Viscosity->Get(i - 1, j, k + 1) +
							Viscosity->Get(i - 1, j, k) +
							Viscosity->Get(i, j, k + 1) +
							Viscosity->Get(i, j, k));
						float viscBack = 0.25f * (Viscosity->Get(i - 1, j, k) +
							Viscosity->Get(i - 1, j, k - 1) +
							Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k - 1));

						float volRight = Volumes.center(i, j, k);
						float volLeft = Volumes.center(i - 1, j, k);
						float volTop = Volumes.edgeW(i, j + 1, k);
						float volBottom = Volumes.edgeW(i, j, k);
						float volFront = Volumes.edgeV(i, j, k + 1);
						float volBack = Volumes.edgeV(i, j, k);

						float factorRight = 2 * factor * viscRight * volRight;
						float factorLeft = 2 * factor * viscLeft * volLeft;
						float factorTop = factor * viscTop * volTop;
						float factorBottom = factor * viscBottom * volBottom;
						float factorFront = factor * viscFront * volFront;
						float factorBack = factor * viscBack * volBack;

						float diag = Volumes.U(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
						matrix.set(row, row, diag);
						if (State.U(i + 1, j, k) == FLUID) { matrix.add(row, mj.U(i + 1, j, k), -factorRight); }
						if (State.U(i - 1, j, k) == FLUID) { matrix.add(row, mj.U(i - 1, j, k), -factorLeft); }
						if (State.U(i, j + 1, k) == FLUID) { matrix.add(row, mj.U(i, j + 1, k), -factorTop); }
						if (State.U(i, j - 1, k) == FLUID) { matrix.add(row, mj.U(i, j - 1, k), -factorBottom); }
						if (State.U(i, j, k + 1) == FLUID) { matrix.add(row, mj.U(i, j, k + 1), -factorFront); }
						if (State.U(i, j, k - 1) == FLUID) { matrix.add(row, mj.U(i, j, k - 1), -factorBack); }

						if (State.V(i, j + 1, k) == FLUID) { matrix.add(row, mj.V(i, j + 1, k), -factorTop); }
						if (State.V(i - 1, j + 1, k) == FLUID) { matrix.add(row, mj.V(i - 1, j + 1, k), factorTop); }
						if (State.V(i, j, k) == FLUID) { matrix.add(row, mj.V(i, j, k), factorBottom); }
						if (State.V(i - 1, j, k) == FLUID) { matrix.add(row, mj.V(i - 1, j, k), -factorBottom); }

						if (State.W(i, j, k + 1) == FLUID) { matrix.add(row, mj.W(i, j, k + 1), -factorFront); }
						if (State.W(i - 1, j, k + 1) == FLUID) { matrix.add(row, mj.W(i - 1, j, k + 1), factorFront); }
						if (State.W(i, j, k) == FLUID) { matrix.add(row, mj.W(i, j, k), factorBack); }
						if (State.W(i - 1, j, k) == FLUID) { matrix.add(row, mj.W(i - 1, j, k), -factorBack); }

						float rval = Volumes.U(i, j, k) * VelocityField->U(i, j, k);
						if (State.U(i + 1, j, k) == SOLID) { rval -= -factorRight * VelocityField->U(i + 1, j, k); }
						if (State.U(i - 1, j, k) == SOLID) { rval -= -factorLeft * VelocityField->U(i - 1, j, k); }
						if (State.U(i, j + 1, k) == SOLID) { rval -= -factorTop * VelocityField->U(i, j + 1, k); }
						if (State.U(i, j - 1, k) == SOLID) { rval -= -factorBottom * VelocityField->U(i, j - 1, k); }
						if (State.U(i, j, k + 1) == SOLID) { rval -= -factorFront * VelocityField->U(i, j, k + 1); }
						if (State.U(i, j, k - 1) == SOLID) { rval -= -factorBack * VelocityField->U(i, j, k - 1); }

						if (State.V(i, j + 1, k) == SOLID) { rval -= -factorTop * VelocityField->V(i, j + 1, k); }
						if (State.V(i - 1, j + 1, k) == SOLID) { rval -= factorTop * VelocityField->V(i - 1, j + 1, k); }
						if (State.V(i, j, k) == SOLID) { rval -= factorBottom * VelocityField->V(i, j, k); }
						if (State.V(i - 1, j, k) == SOLID) { rval -= -factorBottom * VelocityField->V(i - 1, j, k); }

						if (State.W(i, j, k + 1) == SOLID) { rval -= -factorFront * VelocityField->W(i, j, k + 1); }
						if (State.W(i - 1, j, k + 1) == SOLID) { rval -= factorFront * VelocityField->W(i - 1, j, k + 1); }
						if (State.W(i, j, k) == SOLID) { rval -= factorBack * VelocityField->W(i, j, k); }
						if (State.W(i - 1, j, k) == SOLID) { rval -= -factorBack * VelocityField->W(i - 1, j, k); }
						rhs[row] = rval;

					}
				}
			}
		}

		__device__ __host__ void InitializeLinearSystemV(SparseMatrix<float>& matrix, std::vector<float>& rhs) {
			MatrixIndexer& mj = MatrixIndex;
			FaceState FLUID = FaceState::FLUID;
			FaceState SOLID = FaceState::SOLID;

			float invdx = 1.0f / DX;
			float factor = DeltaTime * invdx * invdx;
			for (int k = 1; k < Size.z; k++) {
				for (int j = 1; j < Size.y; j++) {
					for (int i = 1; i < Size.x; i++) {

						if (State.V(i, j, k) != FaceState::FLUID) {
							continue;
						}

						int row = MatrixIndex.V(i, j, k);
						if (row == -1) {
							continue;
						}

						float viscRight = 0.25f * (Viscosity->Get(i, j - 1, k) +
							Viscosity->Get(i + 1, j - 1, k) +
							Viscosity->Get(i, j, k) +
							Viscosity->Get(i + 1, j, k));
						float viscLeft = 0.25f * (Viscosity->Get(i, j - 1, k) +
							Viscosity->Get(i - 1, j - 1, k) +
							Viscosity->Get(i, j, k) +
							Viscosity->Get(i - 1, j, k));

						float viscTop = Viscosity->Get(i, j, k);
						float viscBottom = Viscosity->Get(i, j - 1, k);

						float viscFront = 0.25f * (Viscosity->Get(i, j - 1, k) +
							Viscosity->Get(i, j - 1, k + 1) +
							Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k + 1));
						float viscBack = 0.25f * (Viscosity->Get(i, j - 1, k) +
							Viscosity->Get(i, j - 1, k - 1) +
							Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k - 1));

						float volRight = Volumes.edgeW(i + 1, j, k);
						float volLeft = Volumes.edgeW(i, j, k);
						float volTop = Volumes.center(i, j, k);
						float volBottom = Volumes.center(i, j - 1, k);
						float volFront = Volumes.edgeU(i, j, k + 1);
						float volBack = Volumes.edgeU(i, j, k);

						float factorRight = factor * viscRight * volRight;
						float factorLeft = factor * viscLeft * volLeft;
						float factorTop = 2 * factor * viscTop * volTop;
						float factorBottom = 2 * factor * viscBottom * volBottom;
						float factorFront = factor * viscFront * volFront;
						float factorBack = factor * viscBack * volBack;

						float diag = Volumes.V(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
						matrix.set(row, row, diag);
						if (State.V(i + 1, j, k) == FLUID) { matrix.add(row, mj.V(i + 1, j, k), -factorRight); }
						if (State.V(i - 1, j, k) == FLUID) { matrix.add(row, mj.V(i - 1, j, k), -factorLeft); }
						if (State.V(i, j + 1, k) == FLUID) { matrix.add(row, mj.V(i, j + 1, k), -factorTop); }
						if (State.V(i, j - 1, k) == FLUID) { matrix.add(row, mj.V(i, j - 1, k), -factorBottom); }
						if (State.V(i, j, k + 1) == FLUID) { matrix.add(row, mj.V(i, j, k + 1), -factorFront); }
						if (State.V(i, j, k - 1) == FLUID) { matrix.add(row, mj.V(i, j, k - 1), -factorBack); }

						if (State.U(i + 1, j, k) == FLUID) { matrix.add(row, mj.U(i + 1, j, k), -factorRight); }
						if (State.U(i + 1, j - 1, k) == FLUID) { matrix.add(row, mj.U(i + 1, j - 1, k), factorRight); }
						if (State.U(i, j, k) == FLUID) { matrix.add(row, mj.U(i, j, k), factorLeft); }
						if (State.U(i, j - 1, k) == FLUID) { matrix.add(row, mj.U(i, j - 1, k), -factorLeft); }

						if (State.W(i, j, k + 1) == FLUID) { matrix.add(row, mj.W(i, j, k + 1), -factorFront); }
						if (State.W(i, j - 1, k + 1) == FLUID) { matrix.add(row, mj.W(i, j - 1, k + 1), factorFront); }
						if (State.W(i, j, k) == FLUID) { matrix.add(row, mj.W(i, j, k), factorBack); }
						if (State.W(i, j - 1, k) == FLUID) { matrix.add(row, mj.W(i, j - 1, k), -factorBack); }

						float rval = Volumes.V(i, j, k) * VelocityField->V(i, j, k);
						if (State.V(i + 1, j, k) == SOLID) { rval -= -factorRight * VelocityField->V(i + 1, j, k); }
						if (State.V(i - 1, j, k) == SOLID) { rval -= -factorLeft * VelocityField->V(i - 1, j, k); }
						if (State.V(i, j + 1, k) == SOLID) { rval -= -factorTop * VelocityField->V(i, j + 1, k); }
						if (State.V(i, j - 1, k) == SOLID) { rval -= -factorBottom * VelocityField->V(i, j - 1, k); }
						if (State.V(i, j, k + 1) == SOLID) { rval -= -factorFront * VelocityField->V(i, j, k + 1); }
						if (State.V(i, j, k - 1) == SOLID) { rval -= -factorBack * VelocityField->V(i, j, k - 1); }

						if (State.U(i + 1, j, k) == SOLID) { rval -= -factorRight * VelocityField->U(i + 1, j, k); }
						if (State.U(i + 1, j - 1, k) == SOLID) { rval -= factorRight * VelocityField->U(i + 1, j - 1, k); }
						if (State.U(i, j, k) == SOLID) { rval -= factorLeft * VelocityField->U(i, j, k); }
						if (State.U(i, j - 1, k) == SOLID) { rval -= -factorLeft * VelocityField->U(i, j - 1, k); }

						if (State.W(i, j, k + 1) == SOLID) { rval -= -factorFront * VelocityField->W(i, j, k + 1); }
						if (State.W(i, j - 1, k + 1) == SOLID) { rval -= factorFront * VelocityField->W(i, j - 1, k + 1); }
						if (State.W(i, j, k) == SOLID) { rval -= factorBack * VelocityField->W(i, j, k); }
						if (State.W(i, j - 1, k) == SOLID) { rval -= -factorBack * VelocityField->W(i, j - 1, k); }
						rhs[row] = rval;

					}
				}
			}
		}

		__device__ __host__ void InitializeLinearSystemW(SparseMatrix<float>& matrix, std::vector<float>& rhs) {
			MatrixIndexer& mj = MatrixIndex;
			FaceState FLUID = FaceState::FLUID;
			FaceState SOLID = FaceState::SOLID;

			float invdx = 1.0f / DX;
			float factor = DeltaTime * invdx * invdx;
			for (int k = 1; k < Size.z; k++) {
				for (int j = 1; j < Size.y; j++) {
					for (int i = 1; i < Size.x; i++) {

						if (State.W(i, j, k) != FaceState::FLUID) {
							continue;
						}

						int row = MatrixIndex.W(i, j, k);
						if (row == -1) {
							continue;
						}

						float viscRight = 0.25f * (Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k - 1) +
							Viscosity->Get(i + 1, j, k) +
							Viscosity->Get(i + 1, j, k - 1));
						float viscLeft = 0.25f * (Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k - 1) +
							Viscosity->Get(i - 1, j, k) +
							Viscosity->Get(i - 1, j, k - 1));

						float viscTop = 0.25f * (Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k - 1) +
							Viscosity->Get(i, j + 1, k) +
							Viscosity->Get(i, j + 1, k - 1));
						float viscBottom = 0.25f * (Viscosity->Get(i, j, k) +
							Viscosity->Get(i, j, k - 1) +
							Viscosity->Get(i, j - 1, k) +
							Viscosity->Get(i, j - 1, k - 1));

						float viscFront = Viscosity->Get(i, j, k);
						float viscBack = Viscosity->Get(i, j, k - 1);

						float volRight = Volumes.edgeV(i + 1, j, k);
						float volLeft = Volumes.edgeV(i, j, k);
						float volTop = Volumes.edgeU(i, j + 1, k);
						float volBottom = Volumes.edgeU(i, j, k);
						float volFront = Volumes.center(i, j, k);
						float volBack = Volumes.center(i, j, k - 1);

						float factorRight = factor * viscRight * volRight;
						float factorLeft = factor * viscLeft * volLeft;
						float factorTop = factor * viscTop * volTop;
						float factorBottom = factor * viscBottom * volBottom;
						float factorFront = 2 * factor * viscFront * volFront;
						float factorBack = 2 * factor * viscBack * volBack;

						float diag = Volumes.W(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
						matrix.set(row, row, diag);
						if (State.W(i + 1, j, k) == FLUID) { matrix.add(row, mj.W(i + 1, j, k), -factorRight); }
						if (State.W(i - 1, j, k) == FLUID) { matrix.add(row, mj.W(i - 1, j, k), -factorLeft); }
						if (State.W(i, j + 1, k) == FLUID) { matrix.add(row, mj.W(i, j + 1, k), -factorTop); }
						if (State.W(i, j - 1, k) == FLUID) { matrix.add(row, mj.W(i, j - 1, k), -factorBottom); }
						if (State.W(i, j, k + 1) == FLUID) { matrix.add(row, mj.W(i, j, k + 1), -factorFront); }
						if (State.W(i, j, k - 1) == FLUID) { matrix.add(row, mj.W(i, j, k - 1), -factorBack); }

						if (State.U(i + 1, j, k) == FLUID) { matrix.add(row, mj.U(i + 1, j, k), -factorRight); }
						if (State.U(i + 1, j, k - 1) == FLUID) { matrix.add(row, mj.U(i + 1, j, k - 1), factorRight); }
						if (State.U(i, j, k) == FLUID) { matrix.add(row, mj.U(i, j, k), factorLeft); }
						if (State.U(i, j, k - 1) == FLUID) { matrix.add(row, mj.U(i, j, k - 1), -factorLeft); }

						if (State.V(i, j + 1, k) == FLUID) { matrix.add(row, mj.V(i, j + 1, k), -factorTop); }
						if (State.V(i, j + 1, k - 1) == FLUID) { matrix.add(row, mj.V(i, j + 1, k - 1), factorTop); }
						if (State.V(i, j, k) == FLUID) { matrix.add(row, mj.V(i, j, k), factorBottom); }
						if (State.V(i, j, k - 1) == FLUID) { matrix.add(row, mj.V(i, j, k - 1), -factorBottom); }

						float rval = Volumes.W(i, j, k) * VelocityField->W(i, j, k);
						if (State.W(i + 1, j, k) == SOLID) { rval -= -factorRight * VelocityField->W(i + 1, j, k); }
						if (State.W(i - 1, j, k) == SOLID) { rval -= -factorLeft * VelocityField->W(i - 1, j, k); }
						if (State.W(i, j + 1, k) == SOLID) { rval -= -factorTop * VelocityField->W(i, j + 1, k); }
						if (State.W(i, j - 1, k) == SOLID) { rval -= -factorBottom * VelocityField->W(i, j - 1, k); }
						if (State.W(i, j, k + 1) == SOLID) { rval -= -factorFront * VelocityField->W(i, j, k + 1); }
						if (State.W(i, j, k - 1) == SOLID) { rval -= -factorBack * VelocityField->W(i, j, k - 1); }
						if (State.U(i + 1, j, k) == SOLID) { rval -= -factorRight * VelocityField->U(i + 1, j, k); }
						if (State.U(i + 1, j, k - 1) == SOLID) { rval -= factorRight * VelocityField->U(i + 1, j, k - 1); }
						if (State.U(i, j, k) == SOLID) { rval -= factorLeft * VelocityField->U(i, j, k); }
						if (State.U(i, j, k - 1) == SOLID) { rval -= -factorLeft * VelocityField->U(i, j, k - 1); }
						if (State.V(i, j + 1, k) == SOLID) { rval -= -factorTop * VelocityField->V(i, j + 1, k); }
						if (State.V(i, j + 1, k - 1) == SOLID) { rval -= factorTop * VelocityField->V(i, j + 1, k - 1); }
						if (State.V(i, j, k) == SOLID) { rval -= factorBottom * VelocityField->V(i, j, k); }
						if (State.V(i, j, k - 1) == SOLID) { rval -= -factorBottom * VelocityField->V(i, j, k - 1); }
						rhs[row] = rval;

					}
				}
			}
		}

		__device__ __host__ void Init(ViscositySolverDescription desc) {
			SolverTolerance = 1e-6f;
			AcceptableTolerace = 10.0f;
			MaxSolverIterations = 700;

			Size = desc.velocityField->Size;
			DX = desc.cellWidth;
			DeltaTime = desc.deltaTime;
			VelocityField = desc.velocityField;
			LiquidSDF = desc.liquidSDF;
			SolidSDF = desc.solidSDF;
			Viscosity = desc.viscosity;
		}

		__device__ __host__ void ComputeSolidCenterPhi(Array3D<float>& solidCenterPhi) {
			for (int k = 0; k < solidCenterPhi.Size.z; k++) {
				for (int j = 0; j < solidCenterPhi.Size.y; j++) {
					for (int i = 0; i < solidCenterPhi.Size.x; i++) {
						solidCenterPhi.Set(i, j, k, SolidSDF->GetDistanceAtCellCenter(i, j, k));
					}
				}
			}
		}

		__device__ __host__ void CalculateFaceStateGrid() {
			Array3D<float> solidCenterPhi;
			solidCenterPhi.Init(Size.x, Size.y, Size.z);
			ComputeSolidCenterPhi(solidCenterPhi);

			State.Init(Size.x, Size.y, Size.z);
			for (int k = 0; k < State.U.Size.z; k++) {
				for (int j = 0; j < State.U.Size.y; j++) {
					for (int i = 0; i < State.U.Size.x; i++) {
						bool isEdge = i == 0 || i == State.U.Size.x - 1;;
						if (isEdge || solidCenterPhi(i - 1, j, k) + solidCenterPhi(i, j, k) <= 0) {
							State.U.Set(i, j, k, FaceState::SOLID);
						}
						else {
							State.U.Set(i, j, k, FaceState::FLUID);
						}
					}
				}
			}

			for (int k = 0; k < State.V.Size.z; k++) {
				for (int j = 0; j < State.V.Size.y; j++) {
					for (int i = 0; i < State.V.Size.x; i++) {
						bool isEdge = j == 0 || j == State.V.Size.y - 1;
						if (isEdge || solidCenterPhi(i, j - 1, k) + solidCenterPhi(i, j, k) <= 0) {
							State.V.Set(i, j, k, FaceState::SOLID);
						}
						else {
							State.V.Set(i, j, k, FaceState::FLUID);
						}
					}
				}
			}

			for (int k = 0; k < State.W.Size.z; k++) {
				for (int j = 0; j < State.W.Size.y; j++) {
					for (int i = 0; i < State.W.Size.x; i++) {
						bool isEdge = k == 0 || k == State.W.Size.z - 1;
						if (isEdge || solidCenterPhi(i, j, k - 1) + solidCenterPhi(i, j, k) <= 0) {
							State.W.Set(i, j, k, FaceState::SOLID);
						}
						else {
							State.W.Set(i, j, k, FaceState::FLUID);
						}
					}
				}
			}

			solidCenterPhi.HostFree();
		}

		__device__ __host__ void EstimateVolumeFractions(Array3D<float>& volumes, glm::vec3 centerStart,Array3D<bool>& validCells) {
			Array3D<float> nodalPhi;
			Array3D<bool> isNodalSet;
			nodalPhi.Init(volumes.Size.x + 1, volumes.Size.y + 1, volumes.Size.z + 1);
			isNodalSet.Init(volumes.Size.x + 1, volumes.Size.y + 1, volumes.Size.z + 1, false);

			volumes.Fill(0);
			float hdx = 0.5f * DX;

			for (int k = 0; k < volumes.Size.z; k++) {
				for (int j = 0; j < volumes.Size.y; j++) {
					for (int i = 0; i < volumes.Size.x; i++) {
						if (!validCells(i, j, k)) {
							continue;
						}

						glm::vec3 centre = centerStart + GridIndexToCellCenter(i, j, k, DX);

						if (!isNodalSet(i, j, k)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, -hdx, -hdx));
							nodalPhi.Set(i, j, k, n);
							isNodalSet.Set(i, j, k, true);
						}
						float phi000 = nodalPhi(i, j, k);

						if (!isNodalSet(i, j, k + 1)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, -hdx, hdx));
							nodalPhi.Set(i, j, k + 1, n);
							isNodalSet.Set(i, j, k + 1, true);
						}
						float phi001 = nodalPhi(i, j, k + 1);

						if (!isNodalSet(i, j + 1, k)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, hdx, -hdx));
							nodalPhi.Set(i, j + 1, k, n);
							isNodalSet.Set(i, j + 1, k, true);
						}
						float phi010 = nodalPhi(i, j + 1, k);

						if (!isNodalSet(i, j + 1, k + 1)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, hdx, hdx));
							nodalPhi.Set(i, j + 1, k + 1, n);
							isNodalSet.Set(i, j + 1, k + 1, true);
						}
						float phi011 = nodalPhi(i, j + 1, k + 1);

						if (!isNodalSet(i + 1, j, k)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, -hdx, -hdx));
							nodalPhi.Set(i + 1, j, k, n);
							isNodalSet.Set(i + 1, j, k, true);
						}
						float phi100 = nodalPhi(i + 1, j, k);

						if (!isNodalSet(i + 1, j, k + 1)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, -hdx, hdx));
							nodalPhi.Set(i + 1, j, k + 1, n);
							isNodalSet.Set(i + 1, j, k + 1, true);
						}
						float phi101 = nodalPhi(i + 1, j, k + 1);

						if (!isNodalSet(i + 1, j + 1, k)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, hdx, -hdx));
							nodalPhi.Set(i + 1, j + 1, k, n);
							isNodalSet.Set(i + 1, j + 1, k, true);
						}
						float phi110 = nodalPhi(i + 1, j + 1, k);

						if (!isNodalSet(i + 1, j + 1, k + 1)) {
							float n = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, hdx, hdx));
							nodalPhi.Set(i + 1, j + 1, k + 1, n);
							isNodalSet.Set(i + 1, j + 1, k + 1, true);
						}
						float phi111 = nodalPhi(i + 1, j + 1, k + 1);

						if (phi000 < 0 && phi001 < 0 && phi010 < 0 && phi011 < 0 &&
							phi100 < 0 && phi101 < 0 && phi110 < 0 && phi111 < 0) {
							volumes.Set(i, j, k, 1.0);
						}
						else if (phi000 >= 0 && phi001 >= 0 && phi010 >= 0 && phi011 >= 0 &&
							phi100 >= 0 && phi101 >= 0 && phi110 >= 0 && phi111 >= 0) {
							volumes.Set(i, j, k, 0.0);
						}
						else {
							volumes.Set(i, j, k, LevelSetUtils::VolumeFraction(
								phi000, phi100, phi010, phi110, phi001, phi101, phi011, phi111
							));
						}
					}
				}
			}

			nodalPhi.HostFree();
			isNodalSet.HostFree();
		}

		__device__ __host__ void CalculateVolumeGrid() {
			Volumes.Init(Size.x, Size.y, Size.z);
			Array3D<bool> validCells;
			validCells.Init(Size.x + 1, Size.y + 1, Size.z + 1, false);
			for (int k = 0; k < Size.z; k++) {
				for (int j = 0; j < Size.y; j++) {
					for (int i = 0; i < Size.x; i++) {
						if (LiquidSDF->Get(i, j, k) < 0) {
							validCells.Set(i, j, k, true);
						}
					}
				}
			}

			int layers = 2;
			for (int layer = 0; layer < layers; layer++) {
				glm::ivec3 nbs[6];
				Array3D<bool> tempValid = validCells;
				for (int k = 0; k < Size.z + 1; k++) {
					for (int j = 0; j < Size.y + 1; j++) {
						for (int i = 0; i < Size.x + 1; i++) {
							if (validCells(i, j, k)) {
								GetNeighbourGridIndices6({ i, j, k }, nbs);
								for (int nidx = 0; nidx < 6; nidx++) {
									if (tempValid.IsIndexInRange(nbs[nidx])) {
										tempValid.Set(nbs[nidx], true);
									}
								}
							}
						}
					}
				}

				validCells = tempValid;
			}

			float hdx = (float)(0.5 * DX);
			EstimateVolumeFractions(Volumes.center, glm::vec3(hdx, hdx, hdx), validCells);
			EstimateVolumeFractions(Volumes.U, glm::vec3(0, hdx, hdx), validCells);
			EstimateVolumeFractions(Volumes.V, glm::vec3(hdx, 0, hdx), validCells);
			EstimateVolumeFractions(Volumes.W, glm::vec3(hdx, hdx, 0), validCells);
			EstimateVolumeFractions(Volumes.edgeU, glm::vec3(hdx, 0, 0), validCells);
			EstimateVolumeFractions(Volumes.edgeV, glm::vec3(0, hdx, 0), validCells);
			EstimateVolumeFractions(Volumes.edgeW, glm::vec3(0, 0, hdx), validCells);

			validCells.HostFree();
		}

		__device__ __host__ void CalculateMatrixIndexTable() {
			int dim = (Size.x + 1) * Size.y * Size.z +
				Size.x * (Size.y + 1) * Size.z +
				Size.x * Size.y * (Size.z + 1);
			FaceIndexer fidx;
			fidx.Init(Size.x, Size.y, Size.z);
			std::vector<bool> isIndexInMatrix(dim, false);

			for (int k = 1; k < Size.z; k++) {
				for (int j = 1; j < Size.y; j++) {
					for (int i = 1; i < Size.x; i++) {
						if (State.U(i, j, k) != FaceState::FLUID) {
							continue;
						}

						float v = Volumes.U(i, j, k);
						float vRight = Volumes.center(i, j, k);
						float vLeft = Volumes.center(i - 1, j, k);
						float vTop = Volumes.edgeW(i, j + 1, k);
						float vBottom = Volumes.edgeW(i, j, k);
						float vFront = Volumes.edgeV(i, j, k + 1);
						float vBack = Volumes.edgeV(i, j, k);

						if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
							vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
							int index = fidx.U(i, j, k);
							isIndexInMatrix[index] = true;
						}
					}
				}
			}

			for (int k = 1; k < Size.z; k++) {
				for (int j = 1; j < Size.y; j++) {
					for (int i = 1; i < Size.x; i++) {
						if (State.V(i, j, k) != FaceState::FLUID) {
							continue;
						}

						float v = Volumes.V(i, j, k);
						float vRight = Volumes.edgeW(i + 1, j, k);
						float vLeft = Volumes.edgeW(i, j, k);
						float vTop = Volumes.center(i, j, k);
						float vBottom = Volumes.center(i, j - 1, k);
						float vFront = Volumes.edgeU(i, j, k + 1);
						float vBack = Volumes.edgeU(i, j, k);

						if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
							vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
							int index = fidx.V(i, j, k);
							isIndexInMatrix[index] = true;
						}
					}
				}
			}

			for (int k = 1; k < Size.z; k++) {
				for (int j = 1; j < Size.y; j++) {
					for (int i = 1; i < Size.x; i++) {
						if (State.W(i, j, k) != FaceState::FLUID) {
							continue;
						}

						float v = Volumes.W(i, j, k);
						float vRight = Volumes.edgeV(i + 1, j, k);
						float vLeft = Volumes.edgeV(i, j, k);
						float vTop = Volumes.edgeU(i, j + 1, k);
						float vBottom = Volumes.edgeU(i, j, k);
						float vFront = Volumes.center(i, j, k);
						float vBack = Volumes.center(i, j, k - 1);

						if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
							vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
							int index = fidx.W(i, j, k);
							isIndexInMatrix[index] = true;
						}
					}
				}
			}

			std::vector<int> gridToMatrixIndex(dim, -1);
			int matrixindex = 0;

			for (size_t i = 0; i < isIndexInMatrix.size(); i++) {
				if (isIndexInMatrix[i]) {
					gridToMatrixIndex[i] = matrixindex;
					matrixindex++;
				}
			}

			 MatrixIndex.Init(Size.x, Size.y, Size.z, gridToMatrixIndex);
		}

		__device__ __host__ void HostFree() {
			State.HostFree();
			Volumes.HostFree();
		}

		glm::ivec3 Size;
		float DX;
		float DeltaTime;

		MACVelocityField* VelocityField;
		ParticleLevelSet* LiquidSDF;
		MeshLevelSet* SolidSDF;
		Array3D<float>* Viscosity;

		FaceStateGrid State;
		ViscosityVolumeGrid Volumes;
		MatrixIndexer MatrixIndex;

		float SolverTolerance;
		float AcceptableTolerace;
		int MaxSolverIterations;
	};
}

#endif // !VISCOSITY_SOLVER_CUH
