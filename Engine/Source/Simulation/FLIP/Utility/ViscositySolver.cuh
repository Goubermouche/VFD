#ifndef VISCOSITY_SOLVER_CUH
#define VISCOSITY_SOLVER_CUH

#include "Simulation/FLIP/Utility/LevelsetUtils.cuh"
#include "Simulation/FLIP/Utility/SparseMatrix.cuh"
#include "Simulation/FLIP/Utility/PCGSolver.cuh"
#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/MarkerAndCellVelocityField.cuh"

namespace fe {
    struct ViscositySolverParameters {
        float cellwidth;
        float deltaTime;

        MACVelocityField* velocityField;
        ParticleLevelSet* liquidSDF;
        MeshLevelSet* solidSDF;
        Array3D<float>* viscosity;
    };

    struct ViscosityVolumeGrid {
        glm::ivec3 size;
        Array3D<float> center;
        Array3D<float> U;
        Array3D<float> V;
        Array3D<float> W;
        Array3D<float> edgeU;
        Array3D<float> edgeV;
        Array3D<float> edgeW;

        ViscosityVolumeGrid() {}
        void Init(int i, int j, int k) {
            size = { i, j, k };
            center.Init(i, j, k, 0.0f);
            U.Init(i + 1, j, k, 0.0f);
            V.Init(i, j + 1, k, 0.0f);
            W.Init(i, j, k + 1, 0.0f);
            edgeU.Init(i, j + 1, k + 1, 0.0f);
            edgeV.Init(i + 1, j, k + 1, 0.0f);
            edgeW.Init(i + 1, j + 1, k, 0.0f);
        }
          
        void HostFree() {
            center.HostFree();
            U.HostFree();
            V.HostFree();
            W.HostFree();
            edgeU.HostFree();
            edgeV.HostFree();
            edgeW.HostFree();
        }
    };

    enum class FaceState : char {
        air = 0x00,
        fluid = 0x01,
        solid = 0x02
    };

    struct FaceStateGrid {
        glm::ivec3 size;
        Array3D<FaceState> U;
        Array3D<FaceState> V;
        Array3D<FaceState> W;

        FaceStateGrid() {}
        void Init(int i, int j, int k) {
            size = { i, j, k };
            U.Init(i + 1, j, k, FaceState::air);
            V.Init(i, j + 1, k, FaceState::air);
            W.Init(i, j, k + 1, FaceState::air);
        }

        void HostFree() {
            U.HostFree();
            V.HostFree();
            W.HostFree();
        }
    };

    struct FaceIndexer {
        glm::ivec3 size;

        FaceIndexer() {}
        void Init(int i, int j, int k) 
        {
            size = { i,j,k };
            _voffset = (size.x + 1) * size.y * size.z;
            _woffset = _voffset + size.x * (size.y + 1) * size.z;
        }

        int U(int i, int j, int k) {
            return i + (size.x + 1) * (j + k * size.y);
        }

        int V(int i, int j, int k) {
            return _voffset + i + size.x * (j + k * (size.y + 1));
        }

        int W(int i, int j, int k) {
            return _woffset + i + size.x * (j + k * size.y);
        }

    private:

        int _voffset;
        int _woffset;
    };

    struct MatrixIndexer {
        std::vector<int> indexTable;
        FaceIndexer faceIndexer;
        int matrixSize;

        MatrixIndexer() {}
        void Init(int i, int j, int k, std::vector<int> matrixIndexTable) {
            faceIndexer.Init(i, j, k);
            indexTable = matrixIndexTable;

            int matsize = 0;
            for (size_t i = 0; i < indexTable.size(); i++) {
                if (indexTable[i] != -1) {
                    matsize++;
                }
            }

            matrixSize = matsize;
        }

        int U(int i, int j, int k) {
            return indexTable[faceIndexer.U(i, j, k)];
        }

        int V(int i, int j, int k) {
            return indexTable[faceIndexer.V(i, j, k)];
        }

        int W(int i, int j, int k) {
            return indexTable[faceIndexer.W(i, j, k)];
        }
    };

    struct ViscositySolver {
       __host__ bool ApplyViscosityToVelocityField(ViscositySolverParameters params) {
           Init(params);
           CalculateFaceStateGrid();
           CalculateVolumeGrid();
           CalculatateMatrixIndexTable();

           int matsize = _matrixIndex.matrixSize;
           SparseMatrixd matrix(matsize);
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
           size = params.velocityField->Size;
           _dx = params.cellwidth;
           _deltaTime = params.deltaTime;
           _velocityField = params.velocityField;
           _liquidSDF = params.liquidSDF;
           _solidSDF = params.solidSDF;
           _viscosity = params.viscosity;

           _solverTolerance = 1e-6;
           _acceptableTolerace = 10.0;
           _maxSolverIterations = 700;
       }

       __host__ void CalculateFaceStateGrid() {
           Array3D<float> solidCenterPhi;
           solidCenterPhi.Init(size.x, size.y, size.z);
           CalculateSolidCenterPhi(solidCenterPhi);
           _state.Init(size.x, size.y, size.z);

           for (int k = 0; k < _state.U.Size.z; k++) {
               for (int j = 0; j < _state.U.Size.y; j++) {
                   for (int i = 0; i < _state.U.Size.x; i++) {
                       bool isEdge = i == 0 || i == _state.U.Size.x - 1;;
                       if (isEdge || solidCenterPhi(i - 1, j, k) + solidCenterPhi(i, j, k) <= 0) {
                           _state.U.Set(i, j, k, FaceState::solid);
                       }
                       else {
                           _state.U.Set(i, j, k, FaceState::fluid);
                       }
                   }
               }
           }

           for (int k = 0; k < _state.V.Size.z; k++) {
               for (int j = 0; j < _state.V.Size.y; j++) {
                   for (int i = 0; i < _state.V.Size.x; i++) {
                       bool isEdge = j == 0 || j == _state.V.Size.y - 1;
                       if (isEdge || solidCenterPhi(i, j - 1, k) + solidCenterPhi(i, j, k) <= 0) {
                           _state.V.Set(i, j, k, FaceState::solid);
                       }
                       else {
                           _state.V.Set(i, j, k, FaceState::fluid);
                       }
                   }
               }
           }

           for (int k = 0; k < _state.W.Size.z; k++) {
               for (int j = 0; j < _state.W.Size.y; j++) {
                   for (int i = 0; i < _state.W.Size.x; i++) {
                       bool isEdge = k == 0 || k == _state.W.Size.z - 1;
                       if (isEdge || solidCenterPhi(i, j, k - 1) + solidCenterPhi(i, j, k) <= 0) {
                           _state.W.Set(i, j, k, FaceState::solid);
                       }
                       else {
                           _state.W.Set(i, j, k, FaceState::fluid);
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
                       solidCenterPhi.Set(i, j, k, _solidSDF->GetDistanceAtCellCenter(i, j, k));
                   }
               }
           }
       }

       __host__ void CalculateVolumeGrid() {
           _volumes.Init(size.x, size.y, size.z);
           Array3D<bool> validCells;
           validCells.Init(size.x + 1, size.y + 1, size.z + 1, false);

           for (int k = 0; k < size.z; k++) {
               for (int j = 0; j < size.y; j++) {
                   for (int i = 0; i < size.x; i++) {
                       if (_liquidSDF->Get(i, j, k) < 0) {
                           validCells.Set(i, j, k, true);
                       }
                   }
               }
           }

           int layers = 2;
           for (int layer = 0; layer < layers; layer++) {
               glm::ivec3 nbs[6];
               Array3D<bool> tempValid = validCells;
               for (int k = 0; k < size.z + 1; k++) {
                   for (int j = 0; j < size.y + 1; j++) {
                       for (int i = 0; i < size.x + 1; i++) {
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

           float hdx = (float)(0.5 * _dx);
          EstimateVolumeFractions(_volumes.center, glm::vec3(hdx, hdx, hdx), validCells);
          EstimateVolumeFractions(_volumes.U, glm::vec3(0, hdx, hdx), validCells);
          EstimateVolumeFractions(_volumes.V, glm::vec3(hdx, 0, hdx), validCells);
          EstimateVolumeFractions(_volumes.W, glm::vec3(hdx, hdx, 0), validCells);
          EstimateVolumeFractions(_volumes.edgeU, glm::vec3(hdx, 0, 0), validCells);
          EstimateVolumeFractions(_volumes.edgeV, glm::vec3(0, hdx, 0), validCells);
          EstimateVolumeFractions(_volumes.edgeW, glm::vec3(0, 0, hdx), validCells);

           validCells.HostFree();
       }
       __host__ void EstimateVolumeFractions(Array3D<float>& volumes,
           glm::vec3 centerStart,
           Array3D<bool>& validCells) {
           Array3D<float> nodalPhi; 
           Array3D<bool> isNodalSet; 

           nodalPhi.Init(volumes.Size.x + 1, volumes.Size.y + 1, volumes.Size.z + 1);
           isNodalSet.Init(volumes.Size.x + 1, volumes.Size.y + 1, volumes.Size.z + 1, false);

           volumes.Fill(0);
           float hdx = 0.5f * _dx;
           for (int k = 0; k < volumes.Size.z; k++) {
               for (int j = 0; j < volumes.Size.y; j++) {
                   for (int i = 0; i < volumes.Size.x; i++) {
                       if (!validCells(i, j, k)) {
                           continue;
                       }

                       glm::vec3 centre = centerStart + GridIndexToCellCenter(i, j, k, _dx);

                       if (!isNodalSet(i, j, k)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, -hdx, -hdx));
                           nodalPhi.Set(i, j, k, n);
                           isNodalSet.Set(i, j, k, true);
                       }
                       float phi000 = nodalPhi(i, j, k);

                       if (!isNodalSet(i, j, k + 1)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, -hdx, hdx));
                           nodalPhi.Set(i, j, k + 1, n);
                           isNodalSet.Set(i, j, k + 1, true);
                       }
                       float phi001 = nodalPhi(i, j, k + 1);

                       if (!isNodalSet(i, j + 1, k)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, hdx, -hdx));
                           nodalPhi.Set(i, j + 1, k, n);
                           isNodalSet.Set(i, j + 1, k, true);
                       }
                       float phi010 = nodalPhi(i, j + 1, k);

                       if (!isNodalSet(i, j + 1, k + 1)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(-hdx, hdx, hdx));
                           nodalPhi.Set(i, j + 1, k + 1, n);
                           isNodalSet.Set(i, j + 1, k + 1, true);
                       }
                       float phi011 = nodalPhi(i, j + 1, k + 1);

                       if (!isNodalSet(i + 1, j, k)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, -hdx, -hdx));
                           nodalPhi.Set(i + 1, j, k, n);
                           isNodalSet.Set(i + 1, j, k, true);
                       }
                       float phi100 = nodalPhi(i + 1, j, k);

                       if (!isNodalSet(i + 1, j, k + 1)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, -hdx, hdx));
                           nodalPhi.Set(i + 1, j, k + 1, n);
                           isNodalSet.Set(i + 1, j, k + 1, true);
                       }
                       float phi101 = nodalPhi(i + 1, j, k + 1);

                       if (!isNodalSet(i + 1, j + 1, k)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, hdx, -hdx));
                           nodalPhi.Set(i + 1, j + 1, k, n);
                           isNodalSet.Set(i + 1, j + 1, k, true);
                       }
                       float phi110 = nodalPhi(i + 1, j + 1, k);

                       if (!isNodalSet(i + 1, j + 1, k + 1)) {
                           float n = _liquidSDF->TrilinearInterpolate(centre + glm::vec3(hdx, hdx, hdx));
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
           int dim = (size.x + 1) * size.y * size.z +
               size.x * (size.y + 1) * size.z +
               size.x * size.y * (size.z + 1);
           FaceIndexer fidx; 
           fidx.Init(size.x, size.y, size.z);

           std::vector<bool> isIndexInMatrix(dim, false);
           for (int k = 1; k < size.z; k++) {
               for (int j = 1; j < size.y; j++) {
                   for (int i = 1; i < size.x; i++) {
                       if (_state.U(i, j, k) != FaceState::fluid) {
                           continue;
                       }

                       float v = _volumes.U(i, j, k);
                       float vRight = _volumes.center(i, j, k);
                       float vLeft = _volumes.center(i - 1, j, k);
                       float vTop = _volumes.edgeW(i, j + 1, k);
                       float vBottom = _volumes.edgeW(i, j, k);
                       float vFront = _volumes.edgeV(i, j, k + 1);
                       float vBack = _volumes.edgeV(i, j, k);

                       if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
                           vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
                           int index = fidx.U(i, j, k);
                           isIndexInMatrix[index] = true;
                       }
                   }
               }
           }

           for (int k = 1; k < size.z; k++) {
               for (int j = 1; j < size.y; j++) {
                   for (int i = 1; i < size.x; i++) {
                       if (_state.V(i, j, k) != FaceState::fluid) {
                           continue;
                       }

                       float v = _volumes.V(i, j, k);
                       float vRight = _volumes.edgeW(i + 1, j, k);
                       float vLeft = _volumes.edgeW(i, j, k);
                       float vTop = _volumes.center(i, j, k);
                       float vBottom = _volumes.center(i, j - 1, k);
                       float vFront = _volumes.edgeU(i, j, k + 1);
                       float vBack = _volumes.edgeU(i, j, k);

                       if (v > 0.0 || vRight > 0.0 || vLeft > 0.0 || vTop > 0.0 ||
                           vBottom > 0.0 || vFront > 0.0 || vBack > 0.0) {
                           int index = fidx.V(i, j, k);
                           isIndexInMatrix[index] = true;
                       }
                   }
               }
           }

           for (int k = 1; k < size.z; k++) {
               for (int j = 1; j < size.y; j++) {
                   for (int i = 1; i < size.x; i++) {
                       if (_state.W(i, j, k) != FaceState::fluid) {
                           continue;
                       }

                       float v = _volumes.W(i, j, k);
                       float vRight = _volumes.edgeV(i + 1, j, k);
                       float vLeft = _volumes.edgeV(i, j, k);
                       float vTop = _volumes.edgeU(i, j + 1, k);
                       float vBottom = _volumes.edgeU(i, j, k);
                       float vFront = _volumes.center(i, j, k);
                       float vBack = _volumes.center(i, j, k - 1);

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

           _matrixIndex.Init(size.x, size.y, size.z, gridToMatrixIndex);
       }

       __host__ void InitLinearSystem(SparseMatrixd& matrix, std::vector<double>& rhs) {
           InitLinearSystemU(matrix, rhs);
           InitLinearSystemV(matrix, rhs);
           InitLinearSystemW(matrix, rhs);
       }

       __host__ void InitLinearSystemU(SparseMatrixd& matrix, std::vector<double>& rhs) {
           MatrixIndexer& mj = _matrixIndex;
           FaceState FLUID = FaceState::fluid;
           FaceState SOLID = FaceState::solid;

           float invdx = 1.0f / _dx;
           float factor = _deltaTime * invdx * invdx;
           for (int k = 1; k < size.z; k++) {
               for (int j = 1; j < size.y; j++) {
                   for (int i = 1; i < size.x; i++) {

                       if (_state.U(i, j, k) != FaceState::fluid) {
                           continue;
                       }

                       int row = _matrixIndex.U(i, j, k);
                       if (row == -1) {
                           continue;
                       }

                       float viscRight = _viscosity->Get(i, j, k);
                       float viscLeft = _viscosity->Get(i - 1, j, k);

                       float viscTop = 0.25f * (_viscosity->Get(i - 1, j + 1, k) +
                           _viscosity->Get(i - 1, j, k) +
                           _viscosity->Get(i, j + 1, k) +
                           _viscosity->Get(i, j, k));
                       float viscBottom = 0.25f * (_viscosity->Get(i - 1, j, k) +
                           _viscosity->Get(i - 1, j - 1, k) +
                           _viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j - 1, k));

                       float viscFront = 0.25f * (_viscosity->Get(i - 1, j, k + 1) +
                           _viscosity->Get(i - 1, j, k) +
                           _viscosity->Get(i, j, k + 1) +
                           _viscosity->Get(i, j, k));
                       float viscBack = 0.25f * (_viscosity->Get(i - 1, j, k) +
                           _viscosity->Get(i - 1, j, k - 1) +
                           _viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k - 1));

                       float volRight = _volumes.center(i, j, k);
                       float volLeft = _volumes.center(i - 1, j, k);
                       float volTop = _volumes.edgeW(i, j + 1, k);
                       float volBottom = _volumes.edgeW(i, j, k);
                       float volFront = _volumes.edgeV(i, j, k + 1);
                       float volBack = _volumes.edgeV(i, j, k);

                       float factorRight = 2 * factor * viscRight * volRight;
                       float factorLeft = 2 * factor * viscLeft * volLeft;
                       float factorTop = factor * viscTop * volTop;
                       float factorBottom = factor * viscBottom * volBottom;
                       float factorFront = factor * viscFront * volFront;
                       float factorBack = factor * viscBack * volBack;

                       float diag = _volumes.U(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
                       matrix.set(row, row, diag);
                       if (_state.U(i + 1, j, k) == FLUID) { matrix.add(row, mj.U(i + 1, j, k), -factorRight); }
                       if (_state.U(i - 1, j, k) == FLUID) { matrix.add(row, mj.U(i - 1, j, k), -factorLeft); }
                       if (_state.U(i, j + 1, k) == FLUID) { matrix.add(row, mj.U(i, j + 1, k), -factorTop); }
                       if (_state.U(i, j - 1, k) == FLUID) { matrix.add(row, mj.U(i, j - 1, k), -factorBottom); }
                       if (_state.U(i, j, k + 1) == FLUID) { matrix.add(row, mj.U(i, j, k + 1), -factorFront); }
                       if (_state.U(i, j, k - 1) == FLUID) { matrix.add(row, mj.U(i, j, k - 1), -factorBack); }

                       if (_state.V(i, j + 1, k) == FLUID) { matrix.add(row, mj.V(i, j + 1, k), -factorTop); }
                       if (_state.V(i - 1, j + 1, k) == FLUID) { matrix.add(row, mj.V(i - 1, j + 1, k), factorTop); }
                       if (_state.V(i, j, k) == FLUID) { matrix.add(row, mj.V(i, j, k), factorBottom); }
                       if (_state.V(i - 1, j, k) == FLUID) { matrix.add(row, mj.V(i - 1, j, k), -factorBottom); }

                       if (_state.W(i, j, k + 1) == FLUID) { matrix.add(row, mj.W(i, j, k + 1), -factorFront); }
                       if (_state.W(i - 1, j, k + 1) == FLUID) { matrix.add(row, mj.W(i - 1, j, k + 1), factorFront); }
                       if (_state.W(i, j, k) == FLUID) { matrix.add(row, mj.W(i, j, k), factorBack); }
                       if (_state.W(i - 1, j, k) == FLUID) { matrix.add(row, mj.W(i - 1, j, k), -factorBack); }

                       float rval = _volumes.U(i, j, k) * _velocityField->U(i, j, k);
                       if (_state.U(i + 1, j, k) == SOLID) { rval -= -factorRight * _velocityField->U(i + 1, j, k); }
                       if (_state.U(i - 1, j, k) == SOLID) { rval -= -factorLeft * _velocityField->U(i - 1, j, k); }
                       if (_state.U(i, j + 1, k) == SOLID) { rval -= -factorTop * _velocityField->U(i, j + 1, k); }
                       if (_state.U(i, j - 1, k) == SOLID) { rval -= -factorBottom * _velocityField->U(i, j - 1, k); }
                       if (_state.U(i, j, k + 1) == SOLID) { rval -= -factorFront * _velocityField->U(i, j, k + 1); }
                       if (_state.U(i, j, k - 1) == SOLID) { rval -= -factorBack * _velocityField->U(i, j, k - 1); }

                       if (_state.V(i, j + 1, k) == SOLID) { rval -= -factorTop * _velocityField->V(i, j + 1, k); }
                       if (_state.V(i - 1, j + 1, k) == SOLID) { rval -= factorTop * _velocityField->V(i - 1, j + 1, k); }
                       if (_state.V(i, j, k) == SOLID) { rval -= factorBottom * _velocityField->V(i, j, k); }
                       if (_state.V(i - 1, j, k) == SOLID) { rval -= -factorBottom * _velocityField->V(i - 1, j, k); }

                       if (_state.W(i, j, k + 1) == SOLID) { rval -= -factorFront * _velocityField->W(i, j, k + 1); }
                       if (_state.W(i - 1, j, k + 1) == SOLID) { rval -= factorFront * _velocityField->W(i - 1, j, k + 1); }
                       if (_state.W(i, j, k) == SOLID) { rval -= factorBack * _velocityField->W(i, j, k); }
                       if (_state.W(i - 1, j, k) == SOLID) { rval -= -factorBack * _velocityField->W(i - 1, j, k); }
                       rhs[row] = rval;

                   }
               }
           }
       }

       __host__ void InitLinearSystemV(SparseMatrixd& matrix, std::vector<double>& rhs) {
           MatrixIndexer& mj = _matrixIndex;
           FaceState FLUID = FaceState::fluid;
           FaceState SOLID = FaceState::solid;

           float invdx = 1.0f / _dx;
           float factor = _deltaTime * invdx * invdx;
           for (int k = 1; k < size.z; k++) {
               for (int j = 1; j < size.y; j++) {
                   for (int i = 1; i < size.x; i++) {

                       if (_state.V(i, j, k) != FaceState::fluid) {
                           continue;
                       }

                       int row = _matrixIndex.V(i, j, k);
                       if (row == -1) {
                           continue;
                       }

                       float viscRight = 0.25f * (_viscosity->Get(i, j - 1, k) +
                           _viscosity->Get(i + 1, j - 1, k) +
                           _viscosity->Get(i, j, k) +
                           _viscosity->Get(i + 1, j, k));
                       float viscLeft = 0.25f * (_viscosity->Get(i, j - 1, k) +
                           _viscosity->Get(i - 1, j - 1, k) +
                           _viscosity->Get(i, j, k) +
                           _viscosity->Get(i - 1, j, k));

                       float viscTop = _viscosity->Get(i, j, k);
                       float viscBottom = _viscosity->Get(i, j - 1, k);

                       float viscFront = 0.25f * (_viscosity->Get(i, j - 1, k) +
                           _viscosity->Get(i, j - 1, k + 1) +
                           _viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k + 1));
                       float viscBack = 0.25f * (_viscosity->Get(i, j - 1, k) +
                           _viscosity->Get(i, j - 1, k - 1) +
                           _viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k - 1));

                       float volRight = _volumes.edgeW(i + 1, j, k);
                       float volLeft = _volumes.edgeW(i, j, k);
                       float volTop = _volumes.center(i, j, k);
                       float volBottom = _volumes.center(i, j - 1, k);
                       float volFront = _volumes.edgeU(i, j, k + 1);
                       float volBack = _volumes.edgeU(i, j, k);

                       float factorRight = factor * viscRight * volRight;
                       float factorLeft = factor * viscLeft * volLeft;
                       float factorTop = 2 * factor * viscTop * volTop;
                       float factorBottom = 2 * factor * viscBottom * volBottom;
                       float factorFront = factor * viscFront * volFront;
                       float factorBack = factor * viscBack * volBack;

                       float diag = _volumes.V(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
                       matrix.set(row, row, diag);
                       if (_state.V(i + 1, j, k) == FLUID) { matrix.add(row, mj.V(i + 1, j, k), -factorRight); }
                       if (_state.V(i - 1, j, k) == FLUID) { matrix.add(row, mj.V(i - 1, j, k), -factorLeft); }
                       if (_state.V(i, j + 1, k) == FLUID) { matrix.add(row, mj.V(i, j + 1, k), -factorTop); }
                       if (_state.V(i, j - 1, k) == FLUID) { matrix.add(row, mj.V(i, j - 1, k), -factorBottom); }
                       if (_state.V(i, j, k + 1) == FLUID) { matrix.add(row, mj.V(i, j, k + 1), -factorFront); }
                       if (_state.V(i, j, k - 1) == FLUID) { matrix.add(row, mj.V(i, j, k - 1), -factorBack); }

                       if (_state.U(i + 1, j, k) == FLUID) { matrix.add(row, mj.U(i + 1, j, k), -factorRight); }
                       if (_state.U(i + 1, j - 1, k) == FLUID) { matrix.add(row, mj.U(i + 1, j - 1, k), factorRight); }
                       if (_state.U(i, j, k) == FLUID) { matrix.add(row, mj.U(i, j, k), factorLeft); }
                       if (_state.U(i, j - 1, k) == FLUID) { matrix.add(row, mj.U(i, j - 1, k), -factorLeft); }

                       if (_state.W(i, j, k + 1) == FLUID) { matrix.add(row, mj.W(i, j, k + 1), -factorFront); }
                       if (_state.W(i, j - 1, k + 1) == FLUID) { matrix.add(row, mj.W(i, j - 1, k + 1), factorFront); }
                       if (_state.W(i, j, k) == FLUID) { matrix.add(row, mj.W(i, j, k), factorBack); }
                       if (_state.W(i, j - 1, k) == FLUID) { matrix.add(row, mj.W(i, j - 1, k), -factorBack); }

                       float rval = _volumes.V(i, j, k) * _velocityField->V(i, j, k);
                       if (_state.V(i + 1, j, k) == SOLID) { rval -= -factorRight * _velocityField->V(i + 1, j, k); }
                       if (_state.V(i - 1, j, k) == SOLID) { rval -= -factorLeft * _velocityField->V(i - 1, j, k); }
                       if (_state.V(i, j + 1, k) == SOLID) { rval -= -factorTop * _velocityField->V(i, j + 1, k); }
                       if (_state.V(i, j - 1, k) == SOLID) { rval -= -factorBottom * _velocityField->V(i, j - 1, k); }
                       if (_state.V(i, j, k + 1) == SOLID) { rval -= -factorFront * _velocityField->V(i, j, k + 1); }
                       if (_state.V(i, j, k - 1) == SOLID) { rval -= -factorBack * _velocityField->V(i, j, k - 1); }

                       if (_state.U(i + 1, j, k) == SOLID) { rval -= -factorRight * _velocityField->U(i + 1, j, k); }
                       if (_state.U(i + 1, j - 1, k) == SOLID) { rval -= factorRight * _velocityField->U(i + 1, j - 1, k); }
                       if (_state.U(i, j, k) == SOLID) { rval -= factorLeft * _velocityField->U(i, j, k); }
                       if (_state.U(i, j - 1, k) == SOLID) { rval -= -factorLeft * _velocityField->U(i, j - 1, k); }

                       if (_state.W(i, j, k + 1) == SOLID) { rval -= -factorFront * _velocityField->W(i, j, k + 1); }
                       if (_state.W(i, j - 1, k + 1) == SOLID) { rval -= factorFront * _velocityField->W(i, j - 1, k + 1); }
                       if (_state.W(i, j, k) == SOLID) { rval -= factorBack * _velocityField->W(i, j, k); }
                       if (_state.W(i, j - 1, k) == SOLID) { rval -= -factorBack * _velocityField->W(i, j - 1, k); }
                       rhs[row] = rval;

                   }
               }
           }
       }

       __host__ void InitLinearSystemW(SparseMatrixd& matrix, std::vector<double>& rhs) {
           MatrixIndexer& mj = _matrixIndex;
           FaceState FLUID = FaceState::fluid;
           FaceState SOLID = FaceState::solid;

           float invdx = 1.0f / _dx;
           float factor = _deltaTime * invdx * invdx;
           for (int k = 1; k < size.z; k++) {
               for (int j = 1; j < size.y; j++) {
                   for (int i = 1; i < size.x; i++) {

                       if (_state.W(i, j, k) != FaceState::fluid) {
                           continue;
                       }

                       int row = _matrixIndex.W(i, j, k);
                       if (row == -1) {
                           continue;
                       }

                       float viscRight = 0.25f * (_viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k - 1) +
                           _viscosity->Get(i + 1, j, k) +
                           _viscosity->Get(i + 1, j, k - 1));
                       float viscLeft = 0.25f * (_viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k - 1) +
                           _viscosity->Get(i - 1, j, k) +
                           _viscosity->Get(i - 1, j, k - 1));

                       float viscTop = 0.25f * (_viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k - 1) +
                           _viscosity->Get(i, j + 1, k) +
                           _viscosity->Get(i, j + 1, k - 1));
                       float viscBottom = 0.25f * (_viscosity->Get(i, j, k) +
                           _viscosity->Get(i, j, k - 1) +
                           _viscosity->Get(i, j - 1, k) +
                           _viscosity->Get(i, j - 1, k - 1));

                       float viscFront = _viscosity->Get(i, j, k);
                       float viscBack = _viscosity->Get(i, j, k - 1);

                       float volRight = _volumes.edgeV(i + 1, j, k);
                       float volLeft = _volumes.edgeV(i, j, k);
                       float volTop = _volumes.edgeU(i, j + 1, k);
                       float volBottom = _volumes.edgeU(i, j, k);
                       float volFront = _volumes.center(i, j, k);
                       float volBack = _volumes.center(i, j, k - 1);

                       float factorRight = factor * viscRight * volRight;
                       float factorLeft = factor * viscLeft * volLeft;
                       float factorTop = factor * viscTop * volTop;
                       float factorBottom = factor * viscBottom * volBottom;
                       float factorFront = 2 * factor * viscFront * volFront;
                       float factorBack = 2 * factor * viscBack * volBack;

                       float diag = _volumes.W(i, j, k) + factorRight + factorLeft + factorTop + factorBottom + factorFront + factorBack;
                       matrix.set(row, row, diag);
                       if (_state.W(i + 1, j, k) == FLUID) { matrix.add(row, mj.W(i + 1, j, k), -factorRight); }
                       if (_state.W(i - 1, j, k) == FLUID) { matrix.add(row, mj.W(i - 1, j, k), -factorLeft); }
                       if (_state.W(i, j + 1, k) == FLUID) { matrix.add(row, mj.W(i, j + 1, k), -factorTop); }
                       if (_state.W(i, j - 1, k) == FLUID) { matrix.add(row, mj.W(i, j - 1, k), -factorBottom); }
                       if (_state.W(i, j, k + 1) == FLUID) { matrix.add(row, mj.W(i, j, k + 1), -factorFront); }
                       if (_state.W(i, j, k - 1) == FLUID) { matrix.add(row, mj.W(i, j, k - 1), -factorBack); }

                       if (_state.U(i + 1, j, k) == FLUID) { matrix.add(row, mj.U(i + 1, j, k), -factorRight); }
                       if (_state.U(i + 1, j, k - 1) == FLUID) { matrix.add(row, mj.U(i + 1, j, k - 1), factorRight); }
                       if (_state.U(i, j, k) == FLUID) { matrix.add(row, mj.U(i, j, k), factorLeft); }
                       if (_state.U(i, j, k - 1) == FLUID) { matrix.add(row, mj.U(i, j, k - 1), -factorLeft); }

                       if (_state.V(i, j + 1, k) == FLUID) { matrix.add(row, mj.V(i, j + 1, k), -factorTop); }
                       if (_state.V(i, j + 1, k - 1) == FLUID) { matrix.add(row, mj.V(i, j + 1, k - 1), factorTop); }
                       if (_state.V(i, j, k) == FLUID) { matrix.add(row, mj.V(i, j, k), factorBottom); }
                       if (_state.V(i, j, k - 1) == FLUID) { matrix.add(row, mj.V(i, j, k - 1), -factorBottom); }

                       float rval = _volumes.W(i, j, k) * _velocityField->W(i, j, k);
                       if (_state.W(i + 1, j, k) == SOLID) { rval -= -factorRight * _velocityField->W(i + 1, j, k); }
                       if (_state.W(i - 1, j, k) == SOLID) { rval -= -factorLeft * _velocityField->W(i - 1, j, k); }
                       if (_state.W(i, j + 1, k) == SOLID) { rval -= -factorTop * _velocityField->W(i, j + 1, k); }
                       if (_state.W(i, j - 1, k) == SOLID) { rval -= -factorBottom * _velocityField->W(i, j - 1, k); }
                       if (_state.W(i, j, k + 1) == SOLID) { rval -= -factorFront * _velocityField->W(i, j, k + 1); }
                       if (_state.W(i, j, k - 1) == SOLID) { rval -= -factorBack * _velocityField->W(i, j, k - 1); }
                       if (_state.U(i + 1, j, k) == SOLID) { rval -= -factorRight * _velocityField->U(i + 1, j, k); }
                       if (_state.U(i + 1, j, k - 1) == SOLID) { rval -= factorRight * _velocityField->U(i + 1, j, k - 1); }
                       if (_state.U(i, j, k) == SOLID) { rval -= factorLeft * _velocityField->U(i, j, k); }
                       if (_state.U(i, j, k - 1) == SOLID) { rval -= -factorLeft * _velocityField->U(i, j, k - 1); }
                       if (_state.V(i, j + 1, k) == SOLID) { rval -= -factorTop * _velocityField->V(i, j + 1, k); }
                       if (_state.V(i, j + 1, k - 1) == SOLID) { rval -= factorTop * _velocityField->V(i, j + 1, k - 1); }
                       if (_state.V(i, j, k) == SOLID) { rval -= factorBottom * _velocityField->V(i, j, k); }
                       if (_state.V(i, j, k - 1) == SOLID) { rval -= -factorBottom * _velocityField->V(i, j, k - 1); }
                       rhs[row] = rval;

                   }
               }
           }
       }

       __host__ bool SolveLinearSystem(SparseMatrixd& matrix, std::vector<double>& rhs,
           std::vector<double>& soln) {
           PCGSolver<double> solver;
           solver.setSolverParameters(_solverTolerance, _maxSolverIterations);

           double estimatedError;
           int numIterations;
           bool success = solver.solve(matrix, rhs, soln, estimatedError, numIterations);

           if (success) {
               std::cout << "\n\tViscosity Solver Iterations: " << numIterations <<
                   "\n\tEstimated Error: " << estimatedError << "\n\n";
               return true;
           }
           else if (numIterations == _maxSolverIterations && estimatedError < _acceptableTolerace) {
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

       __host__ void ApplySolutionToVelocityField(std::vector<double>& soln) {
           _velocityField->Clear();
           for (int k = 0; k < size.z; k++) {
               for (int j = 0; j < size.y; j++) {
                   for (int i = 0; i < size.x + 1; i++) {
                       int matidx = _matrixIndex.U(i, j, k);
                       if (matidx != -1) {
                           _velocityField->SetU(i, j, k, soln[matidx]);
                       }
                   }
               }
           }

           for (int k = 0; k < size.z; k++) {
               for (int j = 0; j < size.y + 1; j++) {
                   for (int i = 0; i < size.x; i++) {
                       int matidx = _matrixIndex.V(i, j, k);
                       if (matidx != -1) {
                           _velocityField->SetV(i, j, k, soln[matidx]);
                       }
                   }
               }
           }

           for (int k = 0; k < size.z + 1; k++) {
               for (int j = 0; j < size.y; j++) {
                   for (int i = 0; i < size.x; i++) {
                       int matidx = _matrixIndex.W(i, j, k);
                       if (matidx != -1) {
                           _velocityField->SetW(i, j, k, soln[matidx]);
                       }
                   }
               }
           }
       }

       __host__ void HostFree() {
           _volumes.HostFree();
           _state.HostFree();
       }

       glm::ivec3 size;
       float _dx;
       float _deltaTime;
       MACVelocityField* _velocityField;
       ParticleLevelSet* _liquidSDF;
       MeshLevelSet* _solidSDF;
       Array3D<float>* _viscosity;

       FaceStateGrid _state;
       ViscosityVolumeGrid _volumes;
       MatrixIndexer _matrixIndex;

       double _solverTolerance;
       double _acceptableTolerace;
       int _maxSolverIterations;
    };
}

#endif // !VISCOSITY_SOLVER_CUH