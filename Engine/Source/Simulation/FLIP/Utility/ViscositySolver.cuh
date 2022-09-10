#ifndef VISCOSITY_SOLVER_CUH
#define VISCOSITY_SOLVER_CUH

#include "Simulation/FLIP/Utility/LevelsetUtils.cuh"
#include "Simulation/FLIP/Utility/SparseMatrix.cuh"
#include "Simulation/FLIP/Utility/PCGSolver.cuh"
#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/MACVelocityField.cuh"

namespace fe {
    struct ViscositySolverParameters {
        float CellWidth;
        float DeltaTime;

        MACVelocityField* VelocityField;
        ParticleLevelSet* LiquidSDF;
        MeshLevelSet* SolidSDF;
        Array3D<float>* Viscosity;
    };

    struct ViscosityVolumeGrid {
        glm::ivec3 Size;
        Array3D<float> Center;
        Array3D<float> U;
        Array3D<float> V;
        Array3D<float> W;
        Array3D<float> EdgeU;
        Array3D<float> EdgeV;
        Array3D<float> EdgeW;

        ViscosityVolumeGrid() {}

        void Init(int i, int j, int k) {
            Size = { i, j, k };
            Center.Init(i, j, k, 0.0f);
            U.Init(i + 1, j, k, 0.0f);
            V.Init(i, j + 1, k, 0.0f);
            W.Init(i, j, k + 1, 0.0f);
            EdgeU.Init(i, j + 1, k + 1, 0.0f);
            EdgeV.Init(i + 1, j, k + 1, 0.0f);
            EdgeW.Init(i + 1, j + 1, k, 0.0f);
        }
          
        void HostFree() {
            Center.HostFree();
            U.HostFree();
            V.HostFree();
            W.HostFree();
            EdgeU.HostFree();
            EdgeV.HostFree();
            EdgeW.HostFree();
        }
    };

    enum class FaceState : char {
        Air = 0x00,
        Fluid = 0x01,
        Solid = 0x02
    };

    struct FaceStateGrid {
        glm::ivec3 Size;
        Array3D<FaceState> U;
        Array3D<FaceState> V;
        Array3D<FaceState> W;

        FaceStateGrid() {}
        void Init(int i, int j, int k) {
            Size = { i, j, k };
            U.Init(i + 1, j, k, FaceState::Air);
            V.Init(i, j + 1, k, FaceState::Air);
            W.Init(i, j, k + 1, FaceState::Air);
        }

        void HostFree() {
            U.HostFree();
            V.HostFree();
            W.HostFree();
        }
    };

    struct FaceIndexer {
        glm::ivec3 Size;

        FaceIndexer() {}
        void Init(int i, int j, int k) 
        {
            Size = { i,j,k };
            VOffset = (Size.x + 1) * Size.y * Size.z;
            WOffset = VOffset + Size.x * (Size.y + 1) * Size.z;
        }

        int U(int i, int j, int k) {
            return i + (Size.x + 1) * (j + k * Size.y);
        }

        int V(int i, int j, int k) {
            return VOffset + i + Size.x * (j + k * (Size.y + 1));
        }

        int W(int i, int j, int k) {
            return WOffset + i + Size.x * (j + k * Size.y);
        }
    private:
        int VOffset;
        int WOffset;
    };

    struct MatrixIndexer {
        MatrixIndexer() {}
        void Init(int i, int j, int k, std::vector<int> matrixIndexTable) {
            FaceIndexer.Init(i, j, k);
            IndexTable = matrixIndexTable;

            int matsize = 0;
            for (size_t i = 0; i < IndexTable.size(); i++) {
                if (IndexTable[i] != -1) {
                    matsize++;
                }
            }

            MatrixSize = matsize;
        }

        int U(int i, int j, int k) {
            return IndexTable[FaceIndexer.U(i, j, k)];
        }

        int V(int i, int j, int k) {
            return IndexTable[FaceIndexer.V(i, j, k)];
        }

        int W(int i, int j, int k) {
            return IndexTable[FaceIndexer.W(i, j, k)];
        }

        std::vector<int> IndexTable;
        FaceIndexer FaceIndexer;
        int MatrixSize;
    };

    struct ViscositySolver {
       __host__ bool ApplyViscosityToVelocityField(ViscositySolverParameters params) {
           Init(params);
           CalculateFaceStateGrid();
           CalculateVolumeGrid();
           CalculatateMatrixIndexTable();

           int matsize = MatrixIndex.MatrixSize;
           SparseMatrix<double> matrix(matsize);
           std::vector<double> rhs(matsize, 0);
           std::vector<double> soln(matsize, 0);

           InitLinearSystem(matrix, rhs);
           bool success = SolveLinearSystem(matrix, rhs, soln);
           
           if (!success) {
               return false;
           }

           ApplySolutionToVelocityField(soln);

           return true;
       }

       __host__ void Init(ViscositySolverParameters params) {
           Size = params.VelocityField->Size;
           DX = params.CellWidth;
           DeltaTime = params.DeltaTime;
           VelocityField = params.VelocityField;
           LiquidSDF = params.LiquidSDF;
           SolidSDF = params.SolidSDF;
           Viscosity = params.Viscosity;

           SolverTolerance = 1e-6;
           AcceptableTolerace = 10.0;
           MaxSolverIterations = 700;
       }

       __host__ void CalculateFaceStateGrid() {
           Array3D<float> solidCenterPhi;
           solidCenterPhi.Init(Size.x, Size.y, Size.z);
           CalculateSolidCenterPhi(solidCenterPhi);
           State.Init(Size.x, Size.y, Size.z);

           for (int k = 0; k < State.U.Size.z; k++) {
               for (int j = 0; j < State.U.Size.y; j++) {
                   for (int i = 0; i < State.U.Size.x; i++) {
                       bool isEdge = i == 0 || i == State.U.Size.x - 1;;
                       if (isEdge || solidCenterPhi(i - 1, j, k) + solidCenterPhi(i, j, k) <= 0) {
                           State.U.Set(i, j, k, FaceState::Solid);
                       }
                       else {
                           State.U.Set(i, j, k, FaceState::Fluid);
                       }
                   }
               }
           }

           for (int k = 0; k < State.V.Size.z; k++) {
               for (int j = 0; j < State.V.Size.y; j++) {
                   for (int i = 0; i < State.V.Size.x; i++) {
                       bool isEdge = j == 0 || j == State.V.Size.y - 1;
                       if (isEdge || solidCenterPhi(i, j - 1, k) + solidCenterPhi(i, j, k) <= 0) {
                           State.V.Set(i, j, k, FaceState::Solid);
                       }
                       else {
                           State.V.Set(i, j, k, FaceState::Fluid);
                       }
                   }
               }
           }

           for (int k = 0; k < State.W.Size.z; k++) {
               for (int j = 0; j < State.W.Size.y; j++) {
                   for (int i = 0; i < State.W.Size.x; i++) {
                       bool isEdge = k == 0 || k == State.W.Size.z - 1;
                       if (isEdge || solidCenterPhi(i, j, k - 1) + solidCenterPhi(i, j, k) <= 0) {
                           State.W.Set(i, j, k, FaceState::Solid);
                       }
                       else {
                           State.W.Set(i, j, k, FaceState::Fluid);
                       }
                   }
               }
           }


           solidCenterPhi.HostFree();
       }

       __host__ void CalculateSolidCenterPhi(Array3D<float>& solidCenterPhi) {
           for (int k = 0; k < solidCenterPhi.Size.z; k++) {
               for (int j = 0; j < solidCenterPhi.Size.y; j++) {
                   for (int i = 0; i < solidCenterPhi.Size.x; i++) {
                       solidCenterPhi.Set(i, j, k, SolidSDF->GetDistanceAtCellCenter(i, j, k));
                   }
               }
           }
       }

       __host__ void CalculateVolumeGrid() {
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
          EstimateVolumeFractions(Volumes.Center, glm::vec3(hdx, hdx, hdx), validCells);
          EstimateVolumeFractions(Volumes.U, glm::vec3(0, hdx, hdx), validCells);
          EstimateVolumeFractions(Volumes.V, glm::vec3(hdx, 0, hdx), validCells);
          EstimateVolumeFractions(Volumes.W, glm::vec3(hdx, hdx, 0), validCells);
          EstimateVolumeFractions(Volumes.EdgeU, glm::vec3(hdx, 0, 0), validCells);
          EstimateVolumeFractions(Volumes.EdgeV, glm::vec3(0, hdx, 0), validCells);
          EstimateVolumeFractions(Volumes.EdgeW, glm::vec3(0, 0, hdx), validCells);

           validCells.HostFree();
       }

       __host__ void EstimateVolumeFractions(Array3D<float>& volumes, glm::vec3 centerStart, Array3D<bool>& validCells) {
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
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, -hdx, -hdx));
                           nodalPhi.Set(i, j, k, Count);
                           isNodalSet.Set(i, j, k, true);
                       }
                       float phi000 = nodalPhi(i, j, k);

                       if (!isNodalSet(i, j, k + 1)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, -hdx, hdx));
                           nodalPhi.Set(i, j, k + 1, Count);
                           isNodalSet.Set(i, j, k + 1, true);
                       }
                       float phi001 = nodalPhi(i, j, k + 1);

                       if (!isNodalSet(i, j + 1, k)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, hdx, -hdx));
                           nodalPhi.Set(i, j + 1, k, Count);
                           isNodalSet.Set(i, j + 1, k, true);
                       }
                       float phi010 = nodalPhi(i, j + 1, k);

                       if (!isNodalSet(i, j + 1, k + 1)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, hdx, hdx));
                           nodalPhi.Set(i, j + 1, k + 1, Count);
                           isNodalSet.Set(i, j + 1, k + 1, true);
                       }
                       float phi011 = nodalPhi(i, j + 1, k + 1);

                       if (!isNodalSet(i + 1, j, k)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, -hdx, -hdx));
                           nodalPhi.Set(i + 1, j, k, Count);
                           isNodalSet.Set(i + 1, j, k, true);
                       }
                       float phi100 = nodalPhi(i + 1, j, k);

                       if (!isNodalSet(i + 1, j, k + 1)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, -hdx, hdx));
                           nodalPhi.Set(i + 1, j, k + 1, Count);
                           isNodalSet.Set(i + 1, j, k + 1, true);
                       }
                       float phi101 = nodalPhi(i + 1, j, k + 1);

                       if (!isNodalSet(i + 1, j + 1, k)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, hdx, -hdx));
                           nodalPhi.Set(i + 1, j + 1, k, Count);
                           isNodalSet.Set(i + 1, j + 1, k, true);
                       }
                       float phi110 = nodalPhi(i + 1, j + 1, k);

                       if (!isNodalSet(i + 1, j + 1, k + 1)) {
                           float Count = LiquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, hdx, hdx));
                           nodalPhi.Set(i + 1, j + 1, k + 1, Count);
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
                           volumes.Set(i, j, k, VolumeFraction(
                               phi000, phi100, phi010, phi110, phi001, phi101, phi011, phi111
                           ));
                       }
                   }
               }

           }

           nodalPhi.HostFree();
           isNodalSet.HostFree();
       }

       __host__ void CalculatateMatrixIndexTable() {
           int dim = (Size.x + 1) * Size.y * Size.z +
               Size.x * (Size.y + 1) * Size.z +
               Size.x * Size.y * (Size.z + 1);
           FaceIndexer fidx; 
           fidx.Init(Size.x, Size.y, Size.z);

           std::vector<bool> isIndexInMatrix(dim, false);
           for (int k = 1; k < Size.z; k++) {
               for (int j = 1; j < Size.y; j++) {
                   for (int i = 1; i < Size.x; i++) {
                       if (State.U(i, j, k) != FaceState::Fluid) {
                           continue;
                       }

                       float v = Volumes.U(i, j, k);
                       float vRight = Volumes.Center(i, j, k);
                       float vLeft = Volumes.Center(i - 1, j, k);
                       float vTop = Volumes.EdgeW(i, j + 1, k);
                       float vBottom = Volumes.EdgeW(i, j, k);
                       float vFront = Volumes.EdgeV(i, j, k + 1);
                       float vBack = Volumes.EdgeV(i, j, k);

                       if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
                           vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
                           int Indices = fidx.U(i, j, k);
                           isIndexInMatrix[Indices] = true;
                       }
                   }
               }
           }

           for (int k = 1; k < Size.z; k++) {
               for (int j = 1; j < Size.y; j++) {
                   for (int i = 1; i < Size.x; i++) {
                       if (State.V(i, j, k) != FaceState::Fluid) {
                           continue;
                       }

                       float v = Volumes.V(i, j, k);
                       float vRight = Volumes.EdgeW(i + 1, j, k);
                       float vLeft = Volumes.EdgeW(i, j, k);
                       float vTop = Volumes.Center(i, j, k);
                       float vBottom = Volumes.Center(i, j - 1, k);
                       float vFront = Volumes.EdgeU(i, j, k + 1);
                       float vBack = Volumes.EdgeU(i, j, k);

                       if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
                           vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
                           int Indices = fidx.V(i, j, k);
                           isIndexInMatrix[Indices] = true;
                       }
                   }
               }
           }

           for (int k = 1; k < Size.z; k++) {
               for (int j = 1; j < Size.y; j++) {
                   for (int i = 1; i < Size.x; i++) {
                       if (State.W(i, j, k) != FaceState::Fluid) {
                           continue;
                       }

                       float v = Volumes.W(i, j, k);
                       float vRight = Volumes.EdgeV(i + 1, j, k);
                       float vLeft = Volumes.EdgeV(i, j, k);
                       float vTop = Volumes.EdgeU(i, j + 1, k);
                       float vBottom = Volumes.EdgeU(i, j, k);
                       float vFront = Volumes.Center(i, j, k);
                       float vBack = Volumes.Center(i, j, k - 1);

                       if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
                           vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
                           int Indices = fidx.W(i, j, k);
                           isIndexInMatrix[Indices] = true;
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

       __host__ void InitLinearSystem(SparseMatrix<double>& matrix, std::vector<double>& rhs) {
           InitLinearSystemU(matrix, rhs);
           InitLinearSystemV(matrix, rhs);
           InitLinearSystemW(matrix, rhs);
       }

       __host__ void InitLinearSystemU(SparseMatrix<double>& matrix, std::vector<double>& rhs) {
           MatrixIndexer& mj = MatrixIndex;
           FaceState FLUID = FaceState::Fluid;
           FaceState SOLID = FaceState::Solid;

           float invdx = 1.0f / DX;
           float factor = DeltaTime * invdx * invdx;
           for (int k = 1; k < Size.z; k++) {
               for (int j = 1; j < Size.y; j++) {
                   for (int i = 1; i < Size.x; i++) {

                       if (State.U(i, j, k) != FaceState::Fluid) {
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

                       float volRight = Volumes.Center(i, j, k);
                       float volLeft = Volumes.Center(i - 1, j, k);
                       float volTop = Volumes.EdgeW(i, j + 1, k);
                       float volBottom = Volumes.EdgeW(i, j, k);
                       float volFront = Volumes.EdgeV(i, j, k + 1);
                       float volBack = Volumes.EdgeV(i, j, k);

                       float factorRight = 2 * factor * viscRight * volRight;
                       float factorLeft = 2 * factor * viscLeft * volLeft;
                       float factorTop = factor * viscTop * volTop;
                       float factorBottom = factor * viscBottom * volBottom;
                       float factorFront = factor * viscFront * volFront;
                       float factorBack = factor * viscBack * volBack;

                       float Diagonal = Volumes.U(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
                       matrix.Set(row, row, Diagonal);
                       if (State.U(i + 1, j, k) == FLUID) { matrix.Add(row, mj.U(i + 1, j, k), -factorRight); }
                       if (State.U(i - 1, j, k) == FLUID) { matrix.Add(row, mj.U(i - 1, j, k), -factorLeft); }
                       if (State.U(i, j + 1, k) == FLUID) { matrix.Add(row, mj.U(i, j + 1, k), -factorTop); }
                       if (State.U(i, j - 1, k) == FLUID) { matrix.Add(row, mj.U(i, j - 1, k), -factorBottom); }
                       if (State.U(i, j, k + 1) == FLUID) { matrix.Add(row, mj.U(i, j, k + 1), -factorFront); }
                       if (State.U(i, j, k - 1) == FLUID) { matrix.Add(row, mj.U(i, j, k - 1), -factorBack); }

                       if (State.V(i, j + 1, k) == FLUID) { matrix.Add(row, mj.V(i, j + 1, k), -factorTop); }
                       if (State.V(i - 1, j + 1, k) == FLUID) { matrix.Add(row, mj.V(i - 1, j + 1, k), factorTop); }
                       if (State.V(i, j, k) == FLUID) { matrix.Add(row, mj.V(i, j, k), factorBottom); }
                       if (State.V(i - 1, j, k) == FLUID) { matrix.Add(row, mj.V(i - 1, j, k), -factorBottom); }

                       if (State.W(i, j, k + 1) == FLUID) { matrix.Add(row, mj.W(i, j, k + 1), -factorFront); }
                       if (State.W(i - 1, j, k + 1) == FLUID) { matrix.Add(row, mj.W(i - 1, j, k + 1), factorFront); }
                       if (State.W(i, j, k) == FLUID) { matrix.Add(row, mj.W(i, j, k), factorBack); }
                       if (State.W(i - 1, j, k) == FLUID) { matrix.Add(row, mj.W(i - 1, j, k), -factorBack); }

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

       __host__ void InitLinearSystemV(SparseMatrix<double>& matrix, std::vector<double>& rhs) {
           MatrixIndexer& mj = MatrixIndex;
           FaceState FLUID = FaceState::Fluid;
           FaceState SOLID = FaceState::Solid;

           float invdx = 1.0f / DX;
           float factor = DeltaTime * invdx * invdx;
           for (int k = 1; k < Size.z; k++) {
               for (int j = 1; j < Size.y; j++) {
                   for (int i = 1; i < Size.x; i++) {

                       if (State.V(i, j, k) != FaceState::Fluid) {
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

                       float volRight = Volumes.EdgeW(i + 1, j, k);
                       float volLeft = Volumes.EdgeW(i, j, k);
                       float volTop = Volumes.Center(i, j, k);
                       float volBottom = Volumes.Center(i, j - 1, k);
                       float volFront = Volumes.EdgeU(i, j, k + 1);
                       float volBack = Volumes.EdgeU(i, j, k);

                       float factorRight = factor * viscRight * volRight;
                       float factorLeft = factor * viscLeft * volLeft;
                       float factorTop = 2 * factor * viscTop * volTop;
                       float factorBottom = 2 * factor * viscBottom * volBottom;
                       float factorFront = factor * viscFront * volFront;
                       float factorBack = factor * viscBack * volBack;

                       float Diagonal = Volumes.V(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
                       matrix.Set(row, row, Diagonal);
                       if (State.V(i + 1, j, k) == FLUID) { matrix.Add(row, mj.V(i + 1, j, k), -factorRight); }
                       if (State.V(i - 1, j, k) == FLUID) { matrix.Add(row, mj.V(i - 1, j, k), -factorLeft); }
                       if (State.V(i, j + 1, k) == FLUID) { matrix.Add(row, mj.V(i, j + 1, k), -factorTop); }
                       if (State.V(i, j - 1, k) == FLUID) { matrix.Add(row, mj.V(i, j - 1, k), -factorBottom); }
                       if (State.V(i, j, k + 1) == FLUID) { matrix.Add(row, mj.V(i, j, k + 1), -factorFront); }
                       if (State.V(i, j, k - 1) == FLUID) { matrix.Add(row, mj.V(i, j, k - 1), -factorBack); }

                       if (State.U(i + 1, j, k) == FLUID) { matrix.Add(row, mj.U(i + 1, j, k), -factorRight); }
                       if (State.U(i + 1, j - 1, k) == FLUID) { matrix.Add(row, mj.U(i + 1, j - 1, k), factorRight); }
                       if (State.U(i, j, k) == FLUID) { matrix.Add(row, mj.U(i, j, k), factorLeft); }
                       if (State.U(i, j - 1, k) == FLUID) { matrix.Add(row, mj.U(i, j - 1, k), -factorLeft); }

                       if (State.W(i, j, k + 1) == FLUID) { matrix.Add(row, mj.W(i, j, k + 1), -factorFront); }
                       if (State.W(i, j - 1, k + 1) == FLUID) { matrix.Add(row, mj.W(i, j - 1, k + 1), factorFront); }
                       if (State.W(i, j, k) == FLUID) { matrix.Add(row, mj.W(i, j, k), factorBack); }
                       if (State.W(i, j - 1, k) == FLUID) { matrix.Add(row, mj.W(i, j - 1, k), -factorBack); }

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

       __host__ void InitLinearSystemW(SparseMatrix<double>& matrix, std::vector<double>& rhs) {
           MatrixIndexer& mj = MatrixIndex;
           FaceState FLUID = FaceState::Fluid;
           FaceState SOLID = FaceState::Solid;

           float invdx = 1.0f / DX;
           float factor = DeltaTime * invdx * invdx;
           for (int k = 1; k < Size.z; k++) {
               for (int j = 1; j < Size.y; j++) {
                   for (int i = 1; i < Size.x; i++) {

                       if (State.W(i, j, k) != FaceState::Fluid) {
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

                       float volRight = Volumes.EdgeV(i + 1, j, k);
                       float volLeft = Volumes.EdgeV(i, j, k);
                       float volTop = Volumes.EdgeU(i, j + 1, k);
                       float volBottom = Volumes.EdgeU(i, j, k);
                       float volFront = Volumes.Center(i, j, k);
                       float volBack = Volumes.Center(i, j, k - 1);

                       float factorRight = factor * viscRight * volRight;
                       float factorLeft = factor * viscLeft * volLeft;
                       float factorTop = factor * viscTop * volTop;
                       float factorBottom = factor * viscBottom * volBottom;
                       float factorFront = 2 * factor * viscFront * volFront;
                       float factorBack = 2 * factor * viscBack * volBack;

                       float Diagonal = Volumes.W(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
                       matrix.Set(row, row, Diagonal);
                       if (State.W(i + 1, j, k) == FLUID) { matrix.Add(row, mj.W(i + 1, j, k), -factorRight); }
                       if (State.W(i - 1, j, k) == FLUID) { matrix.Add(row, mj.W(i - 1, j, k), -factorLeft); }
                       if (State.W(i, j + 1, k) == FLUID) { matrix.Add(row, mj.W(i, j + 1, k), -factorTop); }
                       if (State.W(i, j - 1, k) == FLUID) { matrix.Add(row, mj.W(i, j - 1, k), -factorBottom); }
                       if (State.W(i, j, k + 1) == FLUID) { matrix.Add(row, mj.W(i, j, k + 1), -factorFront); }
                       if (State.W(i, j, k - 1) == FLUID) { matrix.Add(row, mj.W(i, j, k - 1), -factorBack); }

                       if (State.U(i + 1, j, k) == FLUID) { matrix.Add(row, mj.U(i + 1, j, k), -factorRight); }
                       if (State.U(i + 1, j, k - 1) == FLUID) { matrix.Add(row, mj.U(i + 1, j, k - 1), factorRight); }
                       if (State.U(i, j, k) == FLUID) { matrix.Add(row, mj.U(i, j, k), factorLeft); }
                       if (State.U(i, j, k - 1) == FLUID) { matrix.Add(row, mj.U(i, j, k - 1), -factorLeft); }

                       if (State.V(i, j + 1, k) == FLUID) { matrix.Add(row, mj.V(i, j + 1, k), -factorTop); }
                       if (State.V(i, j + 1, k - 1) == FLUID) { matrix.Add(row, mj.V(i, j + 1, k - 1), factorTop); }
                       if (State.V(i, j, k) == FLUID) { matrix.Add(row, mj.V(i, j, k), factorBottom); }
                       if (State.V(i, j, k - 1) == FLUID) { matrix.Add(row, mj.V(i, j, k - 1), -factorBottom); }

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

       __host__ bool SolveLinearSystem(SparseMatrix<double>& matrix, std::vector<double>& rhs, std::vector<double>& soln) {
           PCGSolver<double> solver;
           solver.SetSolverParameters(SolverTolerance, MaxSolverIterations);

           double estimatedError;
           int numIterations;
           bool success = solver.Solve(matrix, rhs, soln, estimatedError, numIterations);

           if (success) {
               //std::cout << "\n\tViscosity Solver Iterations: " << numIterations <<
               //    "\n\tEstimated Error: " << estimatedError << "\n\n";
               return true;
           }
           else if (numIterations == MaxSolverIterations && estimatedError < AcceptableTolerace) {
               //std::cout << "\n\tViscosity Solver Iterations: " << numIterations <<
               //    "\n\tEstimated Error: " << estimatedError << "\n\n";
               return true;
           }
           else {
               //std::cout << "\n\t***Viscosity Solver FAILED" <<
               //    "\n\tViscosity Solver Iterations: " << numIterations <<
               //    "\n\tEstimated Error: " << estimatedError << "\n\n";
               return false;
           }
       }

       __host__ void ApplySolutionToVelocityField(std::vector<double>& soln) {
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

       __host__ void HostFree() {
           Volumes.HostFree();
           State.HostFree();
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

       double SolverTolerance;
       double AcceptableTolerace;
       int MaxSolverIterations;
    };
}

#endif // !VISCOSITY_SOLVER_CUH