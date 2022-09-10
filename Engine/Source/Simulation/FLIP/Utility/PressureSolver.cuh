#ifndef PRESSURE_SOLVER_CUH
#define PRESSURE_SOLVER_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/MACVelocityField.cuh"

namespace fe {
    struct WeightGrid {
        __host__ void Init(int i, int j, int k) {
            U.Init(i + 1, j, k, 0.0f);
            V.Init(i, j + 1, k, 0.0f);
            W.Init(i, j, k + 1, 0.0f);
        }

        __host__ void HostFree() {
            U.HostFree();
            V.HostFree();
            W.HostFree();
        }

        Array3D<float> U;
        Array3D<float> V;
        Array3D<float> W;
    };

    class GridIndexKeyMap
    {
    public:
        GridIndexKeyMap(){}
        GridIndexKeyMap(int i, int j, int k) 
        {
            Size = { i, j, k };
            Indices = std::vector<int>(i * j * k, NotFoundValue);
        }
        
        void Clear() {
            for (unsigned int i = 0; i < Indices.size(); i++) {
                Indices[i] = NotFoundValue;
            }
        }

        void Insert(glm::ivec3 g, int key) {
            Insert(g.x, g.y, g.z, key);
        }

        void Insert(int i, int j, int k, int key) {
            int flatidx = GetFlatIndex(i, j, k);
            Indices[flatidx] = key;
        }

        int Find(glm::ivec3 g) {
            return Find(g.x, g.y, g.z);
        }

        int Find(int i, int j, int k) {
            if (Indices.size() == 0) {
                return NotFoundValue;
            }

            int flatidx = GetFlatIndex(i, j, k);
            return Indices[flatidx];
        }
    private:
        inline unsigned int GetFlatIndex(int i, int j, int k) {
            return (unsigned int)i + (unsigned int)Size.x * ((unsigned int)j + (unsigned int)Size.y * (unsigned int)k);
        }

        inline unsigned int GetFlatIndex(glm::ivec3 g) {
            return (unsigned int)g.x + (unsigned int)Size.x * ((unsigned int)g.y + (unsigned int)Size.y * (unsigned int)g.z);
        }
    private:
        glm::ivec3 Size;
        std::vector<int> Indices;
        int NotFoundValue = -1;
    };

    struct PressureSolverParameters {
        double CellWidth;
        double Density;
        double DeltaTime;

        MACVelocityField* VelocityField;
        ParticleLevelSet* LiquidSDF;
        WeightGrid* WeightGrid;
    };

    class VectorXd
    {
    public:
        VectorXd() {}
        VectorXd(int size)
            : Vector(size, 0.0) 
        {}

        VectorXd(int size, double fill) 
            : Vector(size, fill) 
        {}

        VectorXd(VectorXd& vector) {
            Vector.reserve(vector.Size());
            for (unsigned int i = 0; i < vector.Size(); i++) {
                Vector.push_back(vector[i]);
            }
        }

        const double operator [](int i) const {
            return Vector[i];
        }

        double& operator[](int i) {
            return Vector[i];
        }

        inline size_t Size() {
            return Vector.size();
        }

        void Fill(double fill) {
            for (unsigned int i = 0; i < Vector.size(); i++) {
                Vector[i] = fill;
            }
        }

        double Dot(VectorXd& vector) {
            double sum = 0.0;
            for (unsigned int i = 0; i < Vector.size(); i++) {
                sum += Vector[i] * vector.Vector[i];
            }

            return sum;
        }

        double AbsMaxCoeff() {
            double max = -std::numeric_limits<double>::infinity();
            for (unsigned int i = 0; i < Vector.size(); i++) {
                if (fabs(Vector[i]) > max) {
                    max = fabs(Vector[i]);
                }
            }

            return max;
        }
    public:
        std::vector<double> Vector;
    };

    struct MatrixCell {
        float Diagonal;
        glm::vec3 Plus;

        MatrixCell() 
            : Diagonal(0.0f), Plus(0, 0, 0) 
        {}
    };

    class MatrixCoefficients
    {
    public:
        MatrixCoefficients() {}
        MatrixCoefficients(int size)
            : Cells(size, MatrixCell()) 
        {}

        const MatrixCell operator [](int i) const {
            return Cells[i];
        }

        MatrixCell& operator [](int i) {
            return Cells[i];
        }

        inline size_t size() {
            return Cells.size();
        }

        std::vector<MatrixCell> Cells;
    };

    struct PressureSolver {
        __host__ Array3D<float> Solve(PressureSolverParameters params) {
            Init(params);
            InitGridIndexKeyMap();
            VectorXd b(MatSize);
            CalculateNegativeDivergenceVector(b);
            if (b.AbsMaxCoeff() < PressureSolveTolerance) {
                Array3D<float> r;
                r.Init(Size.x, Size.y, Size.z, 0.0);
                return r;
            }

            MatrixCoefficients A(MatSize);
            CalculateMatrixCoefficients(A);

            VectorXd precon(MatSize);
            CalculatePreconditionerVector(A, precon);

            VectorXd pressure(MatSize);
            SolvePressureSystem(A, b, precon, pressure);

            Array3D<float> pressureGrid;
            pressureGrid.Init(Size.x, Size.y, Size.z, 0.0);
            for (int i = 0; i < (int)PressureCells.size(); i++) {
                glm::ivec3 g = PressureCells[i];
                pressureGrid.Set(g, (float)pressure[i]);
            }

            return pressureGrid;
        }

        __host__ inline int GridToVectorIndex(int i, int j, int k) {
            return KeyMap.Find(i, j, k);
        }

        __host__ void ApplyPreconditioner(MatrixCoefficients& A, VectorXd& precon, VectorXd& residual, VectorXd& vect) {
            VectorXd q(MatSize);
            glm::ivec3 g;
            for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
                g = PressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int vidx = GridToVectorIndex(i, j, k);

                int vidx_im1 = KeyMap.Find(i - 1, j, k);
                int vidx_jm1 = KeyMap.Find(i, j - 1, k);
                int vidx_km1 = KeyMap.Find(i, j, k - 1);

                double plusi_im1 = 0.0;
                double precon_im1 = 0.0;
                double q_im1 = 0.0;
                if (vidx_im1 != -1) {
                    plusi_im1 = (double)A[vidx_im1].Plus.x;
                    precon_im1 = precon[vidx_im1];
                    q_im1 = q[vidx_im1];
                }

                double plusj_jm1 = 0.0;
                double precon_jm1 = 0.0;
                double q_jm1 = 0.0;
                if (vidx_jm1 != -1) {
                    plusj_jm1 = (double)A[vidx_jm1].Plus.y;
                    precon_jm1 = precon[vidx_jm1];
                    q_jm1 = q[vidx_jm1];
                }

                double plusk_km1 = 0.0;
                double precon_km1 = 0.0;
                double q_km1 = 0.0;
                if (vidx_km1 != -1) {
                    plusk_km1 = (double)A[vidx_km1].Plus.z;
                    precon_km1 = precon[vidx_km1];
                    q_km1 = q[vidx_km1];
                }

                double t = residual[vidx] - plusi_im1 * precon_im1 * q_im1 -
                    plusj_jm1 * precon_jm1 * q_jm1 -
                    plusk_km1 * precon_km1 * q_km1;

                t = t * precon[vidx];
                q[vidx] = t;
            }

            // Solve transpose(A)*z = q
            for (int idx = (int)PressureCells.size() - 1; idx >= 0; idx--) {
                g = PressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int vidx = GridToVectorIndex(i, j, k);

                int vidx_ip1 = KeyMap.Find(i + 1, j, k);
                int vidx_jp1 = KeyMap.Find(i, j + 1, k);
                int vidx_kp1 = KeyMap.Find(i, j, k + 1);

                double vect_ip1 = vidx_ip1 != -1 ? vect[vidx_ip1] : 0.0;
                double vect_jp1 = vidx_jp1 != -1 ? vect[vidx_jp1] : 0.0;
                double vect_kp1 = vidx_kp1 != -1 ? vect[vidx_kp1] : 0.0;

                double plusi = (double)A[vidx].Plus.x;
                double plusj = (double)A[vidx].Plus.y;
                double plusk = (double)A[vidx].Plus.z;

                double preconval = precon[vidx];
                double t = q[vidx] - plusi * preconval * vect_ip1 -
                    plusj * preconval * vect_jp1 -
                    plusk * preconval * vect_kp1;

                t = t * preconval;
                vect[vidx] = t;
            }
        }

        __host__ void ApplyMatrix(MatrixCoefficients& A, VectorXd& x, VectorXd& result) {
            glm::ivec3 g;
            for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
                g = PressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int ridx = GridToVectorIndex(i, j, k);

                // val = dot product of column vector x and idxth row of matrix A
                double val = 0.0;
                int vidx = GridToVectorIndex(i - 1, j, k);
                if (vidx != -1) { val += x.Vector[vidx] * A[vidx].Plus.x; }

                vidx = GridToVectorIndex(i + 1, j, k);
                if (vidx != -1) { val += x.Vector[vidx] * A[ridx].Plus.x; }

                vidx = GridToVectorIndex(i, j - 1, k);
                if (vidx != -1) { val += x.Vector[vidx] * A[vidx].Plus.y; }

                vidx = GridToVectorIndex(i, j + 1, k);
                if (vidx != -1) { val += x.Vector[vidx] * A[ridx].Plus.y; }

                vidx = GridToVectorIndex(i, j, k - 1);
                if (vidx != -1) { val += x.Vector[vidx] * A[vidx].Plus.z; }

                vidx = GridToVectorIndex(i, j, k + 1);
                if (vidx != -1) { val += x.Vector[vidx] * A[ridx].Plus.z; }

                val += x.Vector[ridx] * A.Cells[ridx].Diagonal;

                result.Vector[ridx] = val;
            }
        }

        __host__ void AddScaledVector(VectorXd& v1, VectorXd& v2, double scale) {
            for (unsigned int idx = 0; idx < v1.Size(); idx++) {
                v1.Vector[idx] += v2.Vector[idx] * scale;
            }
        }

        __host__ void AddScaledVectors(VectorXd& v1, double s1, VectorXd& v2, double s2, VectorXd& result) {
            for (unsigned int idx = 0; idx < v1.Size(); idx++) {
                result.Vector[idx] = v1.Vector[idx] * s1 + v2.Vector[idx] * s2;
            }
        }

        __host__ void SolvePressureSystem(MatrixCoefficients& A, VectorXd& b, VectorXd& precon, VectorXd& pressure) {
            double tol = PressureSolveTolerance;
            if (b.AbsMaxCoeff() < tol) {
                return;
            }

            VectorXd residual(b);
            VectorXd auxillary(MatSize);
            ApplyPreconditioner(A, precon, residual, auxillary);

            VectorXd search(auxillary);

            double alpha = 0.0;
            double beta = 0.0;
            double sigma = auxillary.Dot(residual);
            double sigmaNew = 0.0;
            int iterationNumber = 0;

            while (iterationNumber < MaxCGIterations) {
                ApplyMatrix(A, search, auxillary);
                alpha = sigma / auxillary.Dot(search);
                AddScaledVector(pressure, search, alpha);
                AddScaledVector(residual, auxillary, -alpha);

                if (residual.AbsMaxCoeff() < tol) {
                    //std::cout << "\n\tPressure Solver Iterations: " << iterationNumber <<
                    //    "\n\tEstimated Error: " << residual.absMaxCoeff() << "\n\n";
                    return;
                }

                ApplyPreconditioner(A, precon, residual, auxillary);
                sigmaNew = auxillary.Dot(residual);
                beta = sigmaNew / sigma;
                AddScaledVectors(auxillary, 1.0, search, beta, search);
                sigma = sigmaNew;

                iterationNumber++;
            }

            //std::cout << "\n\tPressure Solver FAILED" <<
            //    "\n\tPressure Solver Iterations: " << iterationNumber <<
            //    "\n\tEstimated Error: " << residual.absMaxCoeff() << "\n\n";
        }

        __host__ void CalculatePreconditionerVector(MatrixCoefficients& A, VectorXd& precon) {
            double tau = 0.97;      // Tuning constant
            double sigma = 0.25;    // safety constant
            glm::ivec3 g;
            for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
                g = PressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int vidx = GridToVectorIndex(i, j, k);

                int vidx_im1 = KeyMap.Find(i - 1, j, k);
                int vidx_jm1 = KeyMap.Find(i, j - 1, k);
                int vidx_km1 = KeyMap.Find(i, j, k - 1);

                double Diagonal = (double)A[vidx].Diagonal;

                double plusi_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].Plus.x : 0.0;
                double plusi_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].Plus.x : 0.0;
                double plusi_km1 = vidx_km1 != -1 ? (double)A[vidx_km1].Plus.x : 0.0;

                double plusj_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].Plus.y : 0.0;
                double plusj_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].Plus.y : 0.0;
                double plusj_km1 = vidx_km1 != -1 ? (double)A[vidx_km1].Plus.y : 0.0;

                double plusk_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].Plus.z : 0.0;
                double plusk_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].Plus.z : 0.0;
                double plusk_km1 = vidx_km1 != -1 ? (double)A[vidx_km1].Plus.z : 0.0;

                double precon_im1 = vidx_im1 != -1 ? precon[vidx_im1] : 0.0;
                double precon_jm1 = vidx_jm1 != -1 ? precon[vidx_jm1] : 0.0;
                double precon_km1 = vidx_km1 != -1 ? precon[vidx_km1] : 0.0;

                double v1 = plusi_im1 * precon_im1;
                double v2 = plusj_jm1 * precon_jm1;
                double v3 = plusk_km1 * precon_km1;
                double v4 = precon_im1 * precon_im1;
                double v5 = precon_jm1 * precon_jm1;
                double v6 = precon_km1 * precon_km1;

                double e = Diagonal - v1 * v1 - v2 * v2 - v3 * v3 -
                    tau * (plusi_im1 * (plusj_im1 + plusk_im1) * v4 +
                        plusj_jm1 * (plusi_jm1 + plusk_jm1) * v5 +
                        plusk_km1 * (plusi_km1 + plusj_km1) * v6);

                if (e < sigma * Diagonal) {
                    e = Diagonal;
                }

                if (fabs(e) > 10e-9) {
                    precon[vidx] = 1.0 / sqrt(e);
                }
            }
        }

        __host__ void CalculateMatrixCoefficients(MatrixCoefficients& A) {
            double scale = DeltaTime / (DX * DX);
            glm::ivec3 g;
            for (int idx = 0; idx < (int)PressureCells.size(); idx++) {
                g = PressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int Indices = GridToVectorIndex(i, j, k);

                //right neighbour
                float term = WeightGrid->U(i + 1, j, k) * (float)scale;
                float phiRight = LiquidSDF->Get(i + 1, j, k);
                if (phiRight < 0) {
                    A[Indices].Diagonal += term;
                    A[Indices].Plus.x -= term;
                }
                else {
                    float theta = fmax(LiquidSDF->GetFaceWeightU(i + 1, j, k), MinFrac);
                    A[Indices].Diagonal += term / theta;
                }

                //left neighbour
                term = WeightGrid->U(i, j, k) * (float)scale;
                float phiLeft = LiquidSDF->Get(i - 1, j, k);
                if (phiLeft < 0) {
                    A[Indices].Diagonal += term;
                }
                else {
                    float theta = fmax(LiquidSDF->GetFaceWeightU(i, j, k), MinFrac);
                    A[Indices].Diagonal += term / theta;
                }

                //top neighbour
                term = WeightGrid->V(i, j + 1, k) * (float)scale;
                float phiTop = LiquidSDF->Get(i, j + 1, k);
                if (phiTop < 0) {
                    A[Indices].Diagonal += term;
                    A[Indices].Plus.y -= term;
                }
                else {
                    float theta = fmax(LiquidSDF->GetFaceWeightV(i, j + 1, k), MinFrac);
                    A[Indices].Diagonal += term / theta;
                }

                //bottom neighbour
                term = WeightGrid->V(i, j, k) * (float)scale;
                float phiBot = LiquidSDF->Get(i, j - 1, k);
                if (phiBot < 0) {
                    A[Indices].Diagonal += term;
                }
                else {
                    float theta = fmax(LiquidSDF->GetFaceWeightV(i, j, k), MinFrac);
                    A[Indices].Diagonal += term / theta;
                }

                //far neighbour
                term = WeightGrid->W(i, j, k + 1) * (float)scale;
                float phiFar = LiquidSDF->Get(i, j, k + 1);
                if (phiFar < 0) {
                    A[Indices].Diagonal += term;
                    A[Indices].Plus.z -= term;
                }
                else {
                    float theta = fmax(LiquidSDF->GetFaceWeightW(i, j, k + 1), MinFrac);
                    A[Indices].Diagonal += term / theta;
                }

                //near neighbour
                term = WeightGrid->W(i, j, k) * (float)scale;
                float phiNear = LiquidSDF->Get(i, j, k - 1);
                if (phiNear < 0) {
                    A[Indices].Diagonal += term;
                }
                else {
                    float theta = fmax(LiquidSDF->GetFaceWeightW(i, j, k), MinFrac);
                    A[Indices].Diagonal += term / theta;
                }
            }
        }

        __host__ void CalculateNegativeDivergenceVector(VectorXd& b) {
            glm::ivec3 g;
            for (int idx = 0; idx < (int)PressureCells.size(); idx++) {
                g = PressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;

                double divergence = 0.0;
                divergence -= WeightGrid->U(i + 1, j, k) * VelocityField->U(i + 1, j, k);
                divergence += WeightGrid->U(i, j, k) * VelocityField->U(i, j, k);
                divergence -= WeightGrid->V(i, j + 1, k) * VelocityField->V(i, j + 1, k);
                divergence += WeightGrid->V(i, j, k) * VelocityField->V(i, j, k);
                divergence -= WeightGrid->W(i, j, k + 1) * VelocityField->W(i, j, k + 1);
                divergence += WeightGrid->W(i, j, k) * VelocityField->W(i, j, k);
                divergence /= DX;

                b[GridToVectorIndex(i, j, k)] = divergence;
            }
        }

        __host__ void InitGridIndexKeyMap() {
            KeyMap = GridIndexKeyMap(Size.x, Size.y, Size.z);
            for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
                KeyMap.Insert(PressureCells[idx], idx);
            }
        }

        __host__ void Init(PressureSolverParameters params) {
            DX = params.CellWidth;
            Density = params.Density;
            DeltaTime = params.DeltaTime;

            VelocityField = params.VelocityField;
            LiquidSDF = params.LiquidSDF;
            WeightGrid = params.WeightGrid;
            Size = params.VelocityField->Size;

            PressureSolveTolerance = 1e-9;
            MaxCGIterations = 200;
            MinFrac = 0.01f;

            PressureCells = std::vector<glm::ivec3>();
            for (int k = 1; k < Size.z - 1; k++) {
                for (int j = 1; j < Size.y - 1; j++) {
                    for (int i = 1; i < Size.x - 1; i++) {
                        if (LiquidSDF->Get(i, j, k) < 0) {
                            PressureCells.push_back(glm::ivec3(i, j, k));
                        }
                    }
                }
            }

            MatSize = (int)PressureCells.size();
        }

        double DX;
        double Density;
        double DeltaTime;
        int MatSize;

        MACVelocityField* VelocityField;
        ParticleLevelSet* LiquidSDF;
        WeightGrid* WeightGrid;

        glm::ivec3 Size;

        std::vector<glm::ivec3> PressureCells;

        double PressureSolveTolerance;
        int MaxCGIterations;
        float MinFrac;

        GridIndexKeyMap KeyMap;
    };
}

#endif // !PRESSURE_SOLVER_CUH
