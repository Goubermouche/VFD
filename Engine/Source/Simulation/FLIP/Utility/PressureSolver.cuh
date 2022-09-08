#ifndef PRESSURE_SOLVER_CUH
#define PRESSURE_SOLVER_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/MarkerAndCellVelocityField.cuh"

namespace fe {
    struct WeightGrid {
        Array3D<float> U;
        Array3D<float> V;
        Array3D<float> W;

        __host__ WeightGrid() {}
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
    };

    class GridIndexKeyMap
    {
    public:
        GridIndexKeyMap(){}
        GridIndexKeyMap(int i, int j, int k) : _isize(i), _jsize(j), _ksize(k) 
        {
            _indices = std::vector<int>(i * j * k, _notFoundValue);
        }
        
        void clear() {
            for (unsigned int i = 0; i < _indices.size(); i++) {
                _indices[i] = _notFoundValue;
            }
        }

        void insert(glm::ivec3 g, int key) {
            insert(g.x, g.y, g.z, key);
        }

        void insert(int i, int j, int k, int key) {
            int flatidx = _getFlatIndex(i, j, k);
            _indices[flatidx] = key;
        }

        int find(glm::ivec3 g) {
            return find(g.x, g.y, g.z);
        }
        int find(int i, int j, int k) {
            if (_indices.size() == 0) {
                return _notFoundValue;
            }

            int flatidx = _getFlatIndex(i, j, k);
            return _indices[flatidx];
        }

    private:

        inline unsigned int _getFlatIndex(int i, int j, int k) {
            return (unsigned int)i + (unsigned int)_isize *
                ((unsigned int)j + (unsigned int)_jsize * (unsigned int)k);
        }

        inline unsigned int _getFlatIndex(glm::ivec3 g) {
            return (unsigned int)g.x + (unsigned int)_isize *
                ((unsigned int)g.y + (unsigned int)_jsize * (unsigned int)g.z);
        }

        int _isize = 0;
        int _jsize = 0;
        int _ksize = 0;

        std::vector<int> _indices;
        int _notFoundValue = -1;

    };

    struct PressureSolverParameters {
        double cellwidth;
        double density;
        double deltaTime;

        MACVelocityField* velocityField;
        ParticleLevelSet* liquidSDF;
        WeightGrid* weightGrid;
    };

    class VectorXd
    {
    public:
        VectorXd() {}
        VectorXd(int size) : _vector(size, 0.0) {
        }
        VectorXd(int size, double fill) : _vector(size, fill) {
        }
        VectorXd(VectorXd& vector) {
            _vector.reserve(vector.size());
            for (unsigned int i = 0; i < vector.size(); i++) {
                _vector.push_back(vector[i]);
            }
        }

        const double operator [](int i) const {
            return _vector[i];
        }

        double& operator[](int i) {
            return _vector[i];
        }

        inline size_t size() {
            return _vector.size();
        }

        void fill(double fill) {
            for (unsigned int i = 0; i < _vector.size(); i++) {
                _vector[i] = fill;
            }
        }
        double dot(VectorXd& vector) {
            double sum = 0.0;
            for (unsigned int i = 0; i < _vector.size(); i++) {
                sum += _vector[i] * vector._vector[i];
            }

            return sum;
        }

        double absMaxCoeff() {
            double max = -std::numeric_limits<double>::infinity();
            for (unsigned int i = 0; i < _vector.size(); i++) {
                if (fabs(_vector[i]) > max) {
                    max = fabs(_vector[i]);
                }
            }

            return max;
        }

        std::vector<double> _vector;

    };

    struct MatrixCell {
        float diag;
        float plusi;
        float plusj;
        float plusk;

        MatrixCell() : diag(0.0f), plusi(0.0f), plusj(0.0f), plusk(0.0f) {}
    };

    class MatrixCoefficients
    {
    public:
        MatrixCoefficients() {}
        MatrixCoefficients(int size) : cells(size, MatrixCell()) {
        }

        const MatrixCell operator [](int i) const {
            return cells[i];
        }
        MatrixCell& operator [](int i) {
            return cells[i];
        }

        inline size_t size() {
            return cells.size();
        }

        std::vector<MatrixCell> cells;
    };

    struct PressureSolver {
        __host__ Array3D<float> solve(PressureSolverParameters params) {
            Init(params);
            InitGridIndexKeyMap();
            VectorXd b(_matSize);
            CalculateNegativeDivergenceVector(b);
            if (b.absMaxCoeff() < _pressureSolveTolerance) {
                Array3D<float> r;
                r.Init(size.x, size.y, size.z, 0.0);
                return r;
            }

            MatrixCoefficients A(_matSize);
            CalculateMatrixCoefficients(A);

            VectorXd precon(_matSize);
            CalculatePreconditionerVector(A, precon);

            VectorXd pressure(_matSize);
            SolvePressureSystem(A, b, precon, pressure);

            Array3D<float> pressureGrid;
            pressureGrid.Init(size.x, size.y, size.z, 0.0);
            for (int i = 0; i < (int)_pressureCells.size(); i++) {
                glm::ivec3 g = _pressureCells[i];
                pressureGrid.Set(g, (float)pressure[i]);
            }

            return pressureGrid;
        }

        inline int GridToVectorIndex(int i, int j, int k) {
            return _keymap.find(i, j, k);
        }

        __host__ void ApplyPreconditioner(MatrixCoefficients& A,
            VectorXd& precon,
            VectorXd& residual,
            VectorXd& vect) {
            // Solve A*q = residual
            VectorXd q(_matSize);
            glm::ivec3 g;
            for (unsigned int idx = 0; idx < _pressureCells.size(); idx++) {
                g = _pressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int vidx = GridToVectorIndex(i, j, k);

                int vidx_im1 = _keymap.find(i - 1, j, k);
                int vidx_jm1 = _keymap.find(i, j - 1, k);
                int vidx_km1 = _keymap.find(i, j, k - 1);

                double plusi_im1 = 0.0;
                double precon_im1 = 0.0;
                double q_im1 = 0.0;
                if (vidx_im1 != -1) {
                    plusi_im1 = (double)A[vidx_im1].plusi;
                    precon_im1 = precon[vidx_im1];
                    q_im1 = q[vidx_im1];
                }

                double plusj_jm1 = 0.0;
                double precon_jm1 = 0.0;
                double q_jm1 = 0.0;
                if (vidx_jm1 != -1) {
                    plusj_jm1 = (double)A[vidx_jm1].plusj;
                    precon_jm1 = precon[vidx_jm1];
                    q_jm1 = q[vidx_jm1];
                }

                double plusk_km1 = 0.0;
                double precon_km1 = 0.0;
                double q_km1 = 0.0;
                if (vidx_km1 != -1) {
                    plusk_km1 = (double)A[vidx_km1].plusk;
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
            for (int idx = (int)_pressureCells.size() - 1; idx >= 0; idx--) {
                g = _pressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int vidx = GridToVectorIndex(i, j, k);

                int vidx_ip1 = _keymap.find(i + 1, j, k);
                int vidx_jp1 = _keymap.find(i, j + 1, k);
                int vidx_kp1 = _keymap.find(i, j, k + 1);

                double vect_ip1 = vidx_ip1 != -1 ? vect[vidx_ip1] : 0.0;
                double vect_jp1 = vidx_jp1 != -1 ? vect[vidx_jp1] : 0.0;
                double vect_kp1 = vidx_kp1 != -1 ? vect[vidx_kp1] : 0.0;

                double plusi = (double)A[vidx].plusi;
                double plusj = (double)A[vidx].plusj;
                double plusk = (double)A[vidx].plusk;

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
            for (unsigned int idx = 0; idx < _pressureCells.size(); idx++) {
                g = _pressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int ridx = GridToVectorIndex(i, j, k);

                // val = dot product of column vector x and idxth row of matrix A
                double val = 0.0;
                int vidx = GridToVectorIndex(i - 1, j, k);
                if (vidx != -1) { val += x._vector[vidx] * A[vidx].plusi; }

                vidx = GridToVectorIndex(i + 1, j, k);
                if (vidx != -1) { val += x._vector[vidx] * A[ridx].plusi; }

                vidx = GridToVectorIndex(i, j - 1, k);
                if (vidx != -1) { val += x._vector[vidx] * A[vidx].plusj; }

                vidx = GridToVectorIndex(i, j + 1, k);
                if (vidx != -1) { val += x._vector[vidx] * A[ridx].plusj; }

                vidx = GridToVectorIndex(i, j, k - 1);
                if (vidx != -1) { val += x._vector[vidx] * A[vidx].plusk; }

                vidx = GridToVectorIndex(i, j, k + 1);
                if (vidx != -1) { val += x._vector[vidx] * A[ridx].plusk; }

                val += x._vector[ridx] * A.cells[ridx].diag;

                result._vector[ridx] = val;
            }
        }

        __host__ void AddScaledVector(VectorXd& v1, VectorXd& v2, double scale) {
            for (unsigned int idx = 0; idx < v1.size(); idx++) {
                v1._vector[idx] += v2._vector[idx] * scale;
            }
        }

        __host__ void AddScaledVectors(VectorXd& v1, double s1,
            VectorXd& v2, double s2,
            VectorXd& result) {
            for (unsigned int idx = 0; idx < v1.size(); idx++) {
                result._vector[idx] = v1._vector[idx] * s1 + v2._vector[idx] * s2;
            }
        }

        __host__ void SolvePressureSystem(MatrixCoefficients& A,
            VectorXd& b,
            VectorXd& precon,
            VectorXd& pressure) {
            double tol = _pressureSolveTolerance;
            if (b.absMaxCoeff() < tol) {
                return;
            }

            VectorXd residual(b);
            VectorXd auxillary(_matSize);
            ApplyPreconditioner(A, precon, residual, auxillary);

            VectorXd search(auxillary);

            double alpha = 0.0;
            double beta = 0.0;
            double sigma = auxillary.dot(residual);
            double sigmaNew = 0.0;
            int iterationNumber = 0;

            while (iterationNumber < _maxCGIterations) {
                ApplyMatrix(A, search, auxillary);
                alpha = sigma / auxillary.dot(search);
                AddScaledVector(pressure, search, alpha);
                AddScaledVector(residual, auxillary, -alpha);

                if (residual.absMaxCoeff() < tol) {
                    std::cout << "\n\tPressure Solver Iterations: " << iterationNumber <<
                        "\n\tEstimated Error: " << residual.absMaxCoeff() << "\n\n";
                    return;
                }

                ApplyPreconditioner(A, precon, residual, auxillary);
                sigmaNew = auxillary.dot(residual);
                beta = sigmaNew / sigma;
                AddScaledVectors(auxillary, 1.0, search, beta, search);
                sigma = sigmaNew;

                iterationNumber++;
            }

            std::cout << "\n\tPressure Solver FAILED" <<
                "\n\tPressure Solver Iterations: " << iterationNumber <<
                "\n\tEstimated Error: " << residual.absMaxCoeff() << "\n\n";
        }

        __host__ void CalculatePreconditionerVector(MatrixCoefficients& A, VectorXd& precon) {
            double tau = 0.97;      // Tuning constant
            double sigma = 0.25;    // safety constant
            glm::ivec3 g;
            for (unsigned int idx = 0; idx < _pressureCells.size(); idx++) {
                g = _pressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int vidx = GridToVectorIndex(i, j, k);

                int vidx_im1 = _keymap.find(i - 1, j, k);
                int vidx_jm1 = _keymap.find(i, j - 1, k);
                int vidx_km1 = _keymap.find(i, j, k - 1);

                double diag = (double)A[vidx].diag;

                double plusi_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].plusi : 0.0;
                double plusi_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].plusi : 0.0;
                double plusi_km1 = vidx_km1 != -1 ? (double)A[vidx_km1].plusi : 0.0;

                double plusj_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].plusj : 0.0;
                double plusj_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].plusj : 0.0;
                double plusj_km1 = vidx_km1 != -1 ? (double)A[vidx_km1].plusj : 0.0;

                double plusk_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].plusk : 0.0;
                double plusk_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].plusk : 0.0;
                double plusk_km1 = vidx_km1 != -1 ? (double)A[vidx_km1].plusk : 0.0;

                double precon_im1 = vidx_im1 != -1 ? precon[vidx_im1] : 0.0;
                double precon_jm1 = vidx_jm1 != -1 ? precon[vidx_jm1] : 0.0;
                double precon_km1 = vidx_km1 != -1 ? precon[vidx_km1] : 0.0;

                double v1 = plusi_im1 * precon_im1;
                double v2 = plusj_jm1 * precon_jm1;
                double v3 = plusk_km1 * precon_km1;
                double v4 = precon_im1 * precon_im1;
                double v5 = precon_jm1 * precon_jm1;
                double v6 = precon_km1 * precon_km1;

                double e = diag - v1 * v1 - v2 * v2 - v3 * v3 -
                    tau * (plusi_im1 * (plusj_im1 + plusk_im1) * v4 +
                        plusj_jm1 * (plusi_jm1 + plusk_jm1) * v5 +
                        plusk_km1 * (plusi_km1 + plusj_km1) * v6);

                if (e < sigma * diag) {
                    e = diag;
                }

                if (fabs(e) > 10e-9) {
                    precon[vidx] = 1.0 / sqrt(e);
                }
            }
        }

        __host__ void CalculateMatrixCoefficients(MatrixCoefficients& A) {
            double scale = _deltaTime / (_dx * _dx);
            glm::ivec3 g;
            for (int idx = 0; idx < (int)_pressureCells.size(); idx++) {
                g = _pressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;
                int index = GridToVectorIndex(i, j, k);

                //right neighbour
                float term = _weightGrid->U(i + 1, j, k) * (float)scale;
                float phiRight = _liquidSDF->Get(i + 1, j, k);
                if (phiRight < 0) {
                    A[index].diag += term;
                    A[index].plusi -= term;
                }
                else {
                    float theta = fmax(_liquidSDF->GetFaceWeightU(i + 1, j, k), _minfrac);
                    A[index].diag += term / theta;
                }

                //left neighbour
                term = _weightGrid->U(i, j, k) * (float)scale;
                float phiLeft = _liquidSDF->Get(i - 1, j, k);
                if (phiLeft < 0) {
                    A[index].diag += term;
                }
                else {
                    float theta = fmax(_liquidSDF->GetFaceWeightU(i, j, k), _minfrac);
                    A[index].diag += term / theta;
                }

                //top neighbour
                term = _weightGrid->V(i, j + 1, k) * (float)scale;
                float phiTop = _liquidSDF->Get(i, j + 1, k);
                if (phiTop < 0) {
                    A[index].diag += term;
                    A[index].plusj -= term;
                }
                else {
                    float theta = fmax(_liquidSDF->GetFaceWeightV(i, j + 1, k), _minfrac);
                    A[index].diag += term / theta;
                }

                //bottom neighbour
                term = _weightGrid->V(i, j, k) * (float)scale;
                float phiBot = _liquidSDF->Get(i, j - 1, k);
                if (phiBot < 0) {
                    A[index].diag += term;
                }
                else {
                    float theta = fmax(_liquidSDF->GetFaceWeightV(i, j, k), _minfrac);
                    A[index].diag += term / theta;
                }

                //far neighbour
                term = _weightGrid->W(i, j, k + 1) * (float)scale;
                float phiFar = _liquidSDF->Get(i, j, k + 1);
                if (phiFar < 0) {
                    A[index].diag += term;
                    A[index].plusk -= term;
                }
                else {
                    float theta = fmax(_liquidSDF->GetFaceWeightW(i, j, k + 1), _minfrac);
                    A[index].diag += term / theta;
                }

                //near neighbour
                term = _weightGrid->W(i, j, k) * (float)scale;
                float phiNear = _liquidSDF->Get(i, j, k - 1);
                if (phiNear < 0) {
                    A[index].diag += term;
                }
                else {
                    float theta = fmax(_liquidSDF->GetFaceWeightW(i, j, k), _minfrac);
                    A[index].diag += term / theta;
                }
            }
        }

        __host__ void CalculateNegativeDivergenceVector(VectorXd& b) {
            glm::ivec3 g;
            for (int idx = 0; idx < (int)_pressureCells.size(); idx++) {
                g = _pressureCells[idx];
                int i = g.x;
                int j = g.y;
                int k = g.z;

                double divergence = 0.0;
                divergence -= _weightGrid->U(i + 1, j, k) * _vField->U(i + 1, j, k);
                divergence += _weightGrid->U(i, j, k) * _vField->U(i, j, k);
                divergence -= _weightGrid->V(i, j + 1, k) * _vField->V(i, j + 1, k);
                divergence += _weightGrid->V(i, j, k) * _vField->V(i, j, k);
                divergence -= _weightGrid->W(i, j, k + 1) * _vField->W(i, j, k + 1);
                divergence += _weightGrid->W(i, j, k) * _vField->W(i, j, k);
                divergence /= _dx;

                b[GridToVectorIndex(i, j, k)] = divergence;
            }
        }

        __host__ void InitGridIndexKeyMap() {
            _keymap = GridIndexKeyMap(size.x, size.y, size.z);
            for (unsigned int idx = 0; idx < _pressureCells.size(); idx++) {
                _keymap.insert(_pressureCells[idx], idx);
            }
        }

        __host__ void Init(PressureSolverParameters params) {
            _dx = params.cellwidth;
            _density = params.density;
            _deltaTime = params.deltaTime;

            _vField = params.velocityField;
            _liquidSDF = params.liquidSDF;
            _weightGrid = params.weightGrid;
            size = params.velocityField->Size;

            _pressureSolveTolerance = 1e-9;
            _maxCGIterations = 200;
            _minfrac = 0.01f;

            _pressureCells = std::vector<glm::ivec3>();
            for (int k = 1; k < size.z - 1; k++) {
                for (int j = 1; j < size.y - 1; j++) {
                    for (int i = 1; i < size.x - 1; i++) {
                        if (_liquidSDF->Get(i, j, k) < 0) {
                            _pressureCells.push_back(glm::ivec3(i, j, k));
                        }
                    }
                }
            }

            _matSize = (int)_pressureCells.size();
        }

        double _dx;
        double _density;
        double _deltaTime;
        int _matSize;

        MACVelocityField* _vField;
        ParticleLevelSet* _liquidSDF;
        WeightGrid* _weightGrid;

        glm::ivec3 size;

        std::vector<glm::ivec3> _pressureCells;

        double _pressureSolveTolerance;
        int _maxCGIterations;
        float _minfrac;

        GridIndexKeyMap _keymap;
    };
}

#endif // !PRESSURE_SOLVER_CUH
