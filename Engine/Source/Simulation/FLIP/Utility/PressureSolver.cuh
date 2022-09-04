#ifndef PRESSURE_SOLVER_CUH
#define PRESSURE_SOLVER_CUH 

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/MarkerAndCellVelocityField.cuh"

namespace fe {
	struct WeightGrid {
		__device__ WeightGrid() {}
		__device__ void Init(int i, int y, int z) {
			U.Init(i + 1, y, z, 0.0f);
			V.Init(i, y + 1, z, 0.0f);
			W.Init(i, y, z + 1, 0.0f);
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
	};

	class GridIndexKeyMap
	{
	public:
		GridIndexKeyMap() {}
		GridIndexKeyMap(int i, int y, int z) {
			_indices = std::vector<int>(i * y * z, _notFoundValue);
		}
		~GridIndexKeyMap() {

		}

		void clear() {
			for (unsigned int i = 0; i < _indices.size(); i++) {
				_indices[i] = _notFoundValue;
			}
		}
		void insert(glm::ivec3 g, int key) {
			insert(g.x, g.y, g.z, key);
		}
		void insert(int i, int y, int z, int key) {
			int flatidx = _getFlatIndex(i, y, z);
			_indices[flatidx] = key;
		}
		int find(glm::ivec3 g) {
			return find(g.x, g.y, g.z);
		}
		int find(int i, int y, int z) {
			if (_indices.size() == 0) {
				return _notFoundValue;
			}

			int flatidx = _getFlatIndex(i, y, z);
			return _indices[flatidx];
		}
	private:

		inline unsigned int _getFlatIndex(int i, int y, int z) {
			return (unsigned int)i + (unsigned int)_isize *
				((unsigned int)y + (unsigned int)_jsize * (unsigned int)z);
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

	class VectorXd
	{
	public:
		VectorXd() {}
		VectorXd(int size) : _vector(size, 0.0) {

		}
		VectorXd(int size, float fill) : _vector(size, fill) {
		}
		VectorXd(VectorXd& vector) {
			_vector.reserve(vector.size());
			for (unsigned int i = 0; i < vector.size(); i++) {
				_vector.push_back(vector[i]);
			}
		}

		const float operator [](int i) const {
			return _vector[i];
		}
		float& operator[](int i) {
			return _vector[i];
		}

		inline size_t size() {
			return _vector.size();
		}

		void fill(float fill) {
			for (unsigned int i = 0; i < _vector.size(); i++) {
				_vector[i] = fill;
			}
		}
		float dot(VectorXd& vector) {
			float sum = 0.0;
			for (unsigned int i = 0; i < _vector.size(); i++) {
				sum += _vector[i] * vector._vector[i];
			}

			return sum;
		}
		float absMaxCoeff() {
			float max = -std::numeric_limits<float>::infinity();
			for (unsigned int i = 0; i < _vector.size(); i++) {
				if (fabs(_vector[i]) > max) {
					max = fabs(_vector[i]);
				}
			}

			return max;
		}

		std::vector<float> _vector;
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
		MatrixCoefficients();
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

	struct PressureSolverDescription {
		float cellWidth;
		float density;
		float deltaTime;

		MACVelocityField* velocityField;
		ParticleLevelSet* liquidSDF;
		WeightGrid* weightGrid;
	};

	struct PressureSolver {
		__device__ __host__ PressureSolver() {}

		__device__ __host__ Array3D<float> Solve(PressureSolverDescription desc) {
			Init(desc);
			InitGridIndexKeyMap();

			VectorXd b(MatSize);
			CalculateNegativeDivergenceVector(b);
			if (b.absMaxCoeff() < PressureSolveTolerance) {
				Array3D<float> r;
				r.Init(Size.x, Size.y, Size.z, 0.0f);
				return r;
			}

			MatrixCoefficients A(MatSize);
			CalculateMatrixCoefficients(A);

			VectorXd precon(MatSize);
			CalculatePreconditionerVector(A, precon);

			VectorXd pressure(MatSize);
			SolvePressureSystem(A, b, precon, pressure);

			Array3D<float> pressureGrid;
			pressureGrid.Init(Size.x, Size.y, Size.z, 0.0f);

			for (int i = 0; i < (int)PressureCells.size(); i++) {
				glm::ivec3 g = PressureCells[i];
				pressureGrid.Set(g, (float)pressure[i]);
			}

			return pressureGrid;
		}

		__device__ __host__ int _GridToVectorIndex(glm::ivec3 g) {
			return KeyMap.find(g);
		}

		__device__ __host__ int _GridToVectorIndex(int i, int y, int z) {
			return KeyMap.find(i, y, z);
		}

		__device__ __host__ void SolvePressureSystem(MatrixCoefficients& A,
			VectorXd& b,
			VectorXd& precon,
			VectorXd& pressure) {

			float tol = PressureSolveTolerance;
			if (b.absMaxCoeff() < tol) {
				return;
			}

			VectorXd residual(b);
			VectorXd auxillary(MatSize);
			ApplyPreconditioner(A, precon, residual, auxillary);

			VectorXd search(auxillary);

			float alpha = 0.0;
			float beta = 0.0;
			float sigma = auxillary.dot(residual);
			float sigmaNew = 0.0;
			int iterationNumber = 0;

			while (iterationNumber < maxCGIIterations) {
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

		__device__ __host__ void AddScaledVector(VectorXd& v1, VectorXd& v2, float scale) {
			for (unsigned int idx = 0; idx < v1.size(); idx++) {
				v1._vector[idx] += v2._vector[idx] * scale;
			}
		}

		// result = v1*s1 + v2*s2
		__device__ __host__ void AddScaledVectors(VectorXd& v1, float s1,
			VectorXd& v2, float s2,
			VectorXd& result) {
			for (unsigned int idx = 0; idx < v1.size(); idx++) {
				result._vector[idx] = v1._vector[idx] * s1 + v2._vector[idx] * s2;
			}
		}

		__device__ __host__ void ApplyMatrix(MatrixCoefficients& A, VectorXd& x, VectorXd& result) {
			glm::ivec3 g;
			for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
				g = PressureCells[idx];
				int i = g.x;
				int j = g.y;
				int k = g.z;
				int ridx = _GridToVectorIndex(i, j, k);

				// val = dot product of column vector x and idxth row of matrix A
				float val = 0.0;
				int vidx = _GridToVectorIndex(i - 1, j, k);
				if (vidx != -1) { val += x._vector[vidx] * A[vidx].plusi; }

				vidx = _GridToVectorIndex(i + 1, j, k);
				if (vidx != -1) { val += x._vector[vidx] * A[ridx].plusi; }

				vidx = _GridToVectorIndex(i, j - 1, k);
				if (vidx != -1) { val += x._vector[vidx] * A[vidx].plusj; }

				vidx = _GridToVectorIndex(i, j + 1, k);
				if (vidx != -1) { val += x._vector[vidx] * A[ridx].plusj; }

				vidx = _GridToVectorIndex(i, j, k - 1);
				if (vidx != -1) { val += x._vector[vidx] * A[vidx].plusk; }

				vidx = _GridToVectorIndex(i, j, k + 1);
				if (vidx != -1) { val += x._vector[vidx] * A[ridx].plusk; }

				val += x._vector[ridx] * A.cells[ridx].diag;

				result._vector[ridx] = val;
			}
		}

		__device__ __host__ void ApplyPreconditioner(MatrixCoefficients& A,
			VectorXd& precon,
			VectorXd& residual,
			VectorXd& vect) {

			// Solve A*q = residual
			VectorXd q(MatSize);
			glm::ivec3 g;
			for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
				g = PressureCells[idx];
				int i = g.x;
				int j = g.y;
				int k = g.z;
				int vidx = _GridToVectorIndex(i, j, k);

				int vidx_im1 = KeyMap.find(i - 1, j, k);
				int vidx_jm1 = KeyMap.find(i, j - 1, k);
				int vidx_km1 = KeyMap.find(i, j, k - 1);

				float plusi_im1 = 0.0;
				float precon_im1 = 0.0;
				float q_im1 = 0.0;
				if (vidx_im1 != -1) {
					plusi_im1 = (float)A[vidx_im1].plusi;
					precon_im1 = precon[vidx_im1];
					q_im1 = q[vidx_im1];
				}

				float plusj_jm1 = 0.0;
				float precon_jm1 = 0.0;
				float q_jm1 = 0.0;
				if (vidx_jm1 != -1) {
					plusj_jm1 = (float)A[vidx_jm1].plusj;
					precon_jm1 = precon[vidx_jm1];
					q_jm1 = q[vidx_jm1];
				}

				float plusk_km1 = 0.0;
				float precon_km1 = 0.0;
				float q_km1 = 0.0;
				if (vidx_km1 != -1) {
					plusk_km1 = (float)A[vidx_km1].plusk;
					precon_km1 = precon[vidx_km1];
					q_km1 = q[vidx_km1];
				}

				float t = residual[vidx] - plusi_im1 * precon_im1 * q_im1 -
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
				int vidx = _GridToVectorIndex(i, j, k);

				int vidx_ip1 = KeyMap.find(i + 1, j, k);
				int vidx_jp1 = KeyMap.find(i, j + 1, k);
				int vidx_kp1 = KeyMap.find(i, j, k + 1);

				float vect_ip1 = vidx_ip1 != -1 ? vect[vidx_ip1] : 0.0;
				float vect_jp1 = vidx_jp1 != -1 ? vect[vidx_jp1] : 0.0;
				float vect_kp1 = vidx_kp1 != -1 ? vect[vidx_kp1] : 0.0;

				float plusi = (float)A[vidx].plusi;
				float plusj = (float)A[vidx].plusj;
				float plusk = (float)A[vidx].plusk;

				float preconval = precon[vidx];
				float t = q[vidx] - plusi * preconval * vect_ip1 -
					plusj * preconval * vect_jp1 -
					plusk * preconval * vect_kp1;

				t = t * preconval;
				vect[vidx] = t;
			}
		}

		__device__ __host__ void CalculatePreconditionerVector(MatrixCoefficients& A, VectorXd& precon) {
			float tau = 0.97;      // Tuning constant
			float sigma = 0.25;    // safety constant
			glm::vec3 g;
			for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
				g = PressureCells[idx];
				int i = g.x;
				int j = g.y;
				int k = g.z;
				int vidx = _GridToVectorIndex(i, j, k);

				int vidx_im1 = KeyMap.find(i - 1, j, k);
				int vidx_jm1 = KeyMap.find(i, j - 1, k);
				int vidx_km1 = KeyMap.find(i, j, k - 1);

				float diag = (float)A[vidx].diag;

				float plusi_im1 = vidx_im1 != -1 ? (float)A[vidx_im1].plusi : 0.0;
				float plusi_jm1 = vidx_jm1 != -1 ? (float)A[vidx_jm1].plusi : 0.0;
				float plusi_km1 = vidx_km1 != -1 ? (float)A[vidx_km1].plusi : 0.0;

				float plusj_im1 = vidx_im1 != -1 ? (float)A[vidx_im1].plusj : 0.0;
				float plusj_jm1 = vidx_jm1 != -1 ? (float)A[vidx_jm1].plusj : 0.0;
				float plusj_km1 = vidx_km1 != -1 ? (float)A[vidx_km1].plusj : 0.0;

				float plusk_im1 = vidx_im1 != -1 ? (float)A[vidx_im1].plusk : 0.0;
				float plusk_jm1 = vidx_jm1 != -1 ? (float)A[vidx_jm1].plusk : 0.0;
				float plusk_km1 = vidx_km1 != -1 ? (float)A[vidx_km1].plusk : 0.0;

				float precon_im1 = vidx_im1 != -1 ? precon[vidx_im1] : 0.0;
				float precon_jm1 = vidx_jm1 != -1 ? precon[vidx_jm1] : 0.0;
				float precon_km1 = vidx_km1 != -1 ? precon[vidx_km1] : 0.0;

				float v1 = plusi_im1 * precon_im1;
				float v2 = plusj_jm1 * precon_jm1;
				float v3 = plusk_km1 * precon_km1;
				float v4 = precon_im1 * precon_im1;
				float v5 = precon_jm1 * precon_jm1;
				float v6 = precon_km1 * precon_km1;

				float e = diag - v1 * v1 - v2 * v2 - v3 * v3 -
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

		__device__ __host__ void CalculateMatrixCoefficients(MatrixCoefficients& A) {
			float scale = DeltaTime / (DX * DX);
			glm::ivec3 g;

			for (int idx = 0; idx < (int)PressureCells.size(); idx++) {
				g = PressureCells[idx];
				int x = g.x;
				int y = g.y;
				int z = g.z;
				int index = _GridToVectorIndex(x, y, z);

				//right neighbour
				float term = WeightGrid->U(x + 1, y, z) * (float)scale;
				float phiRight = LiquidSDF->Get(x + 1, y, z);
				if (phiRight < 0) {
					A[index].diag += term;
					A[index].plusi -= term;
				}
				else {
					float theta = fmax(LiquidSDF->GetFaceWeightU(x + 1, y, z), MinFrac);
					A[index].diag += term / theta;
				}

				//left neighbour
				term = WeightGrid->U(x, y, z) * (float)scale;
				float phiLeft = LiquidSDF->Get(x - 1, y, z);
				if (phiLeft < 0) {
					A[index].diag += term;
				}
				else {
					float theta = fmax(LiquidSDF->GetFaceWeightU(x, y, z), MinFrac);
					A[index].diag += term / theta;
				}

				//top neighbour
				term = WeightGrid->V(x, y + 1, z) * (float)scale;
				float phiTop = LiquidSDF->Get(x, y + 1, z);
				if (phiTop < 0) {
					A[index].diag += term;
					A[index].plusj -= term;
				}
				else {
					float theta = fmax(LiquidSDF->GetFaceWeightV(x, y + 1, z), MinFrac);
					A[index].diag += term / theta;
				}

				//bottom neighbour
				term = WeightGrid->V(x, y, z) * (float)scale;
				float phiBot = LiquidSDF->Get(x, y - 1, z);
				if (phiBot < 0) {
					A[index].diag += term;
				}
				else {
					float theta = fmax(LiquidSDF->GetFaceWeightV(x, y, z), MinFrac);
					A[index].diag += term / theta;
				}

				//far neighbour
				term = WeightGrid->W(x, y, z + 1) * (float)scale;
				float phiFar = LiquidSDF->Get(x, y, z + 1);
				if (phiFar < 0) {
					A[index].diag += term;
					A[index].plusk -= term;
				}
				else {
					float theta = fmax(LiquidSDF->GetFaceWeightW(x, y, z + 1), MinFrac);
					A[index].diag += term / theta;
				}

				//near neighbour
				term = WeightGrid->W(x, y, z) * (float)scale;
				float phiNear = LiquidSDF->Get(x, y, z - 1);
				if (phiNear < 0) {
					A[index].diag += term;
				}
				else {
					float theta = fmax(LiquidSDF->GetFaceWeightW(x, y, z), MinFrac);
					A[index].diag += term / theta;
				}
			}
		}

		__device__ __host__ void CalculateNegativeDivergenceVector(VectorXd& b) {
			glm::ivec3 g;
			for (int idx = 0; idx < (int)PressureCells.size(); idx++) {
				g = PressureCells[idx];
				int i = g.x;
				int y = g.y;
				int z = g.z;

				float divergence = 0.0;
				divergence -= WeightGrid->U(i + 1, y, z) * VelocityField->U(i + 1, y, z);
				divergence += WeightGrid->U(i, y, z) * VelocityField->U(i, y, z);
				divergence -= WeightGrid->V(i, y + 1, z) * VelocityField->V(i, y + 1, z);
				divergence += WeightGrid->V(i, y, z) * VelocityField->V(i, y, z);
				divergence -= WeightGrid->W(i, y, z + 1) * VelocityField->W(i, y, z + 1);
				divergence += WeightGrid->W(i, y, z) * VelocityField->W(i, y, z);
				divergence /= DX;

				b[_GridToVectorIndex(i, y, z)] = divergence;
			}
		}

		__device__ __host__ void Init(PressureSolverDescription desc) {
			PressureSolveTolerance = 1e-9;
			maxCGIIterations = 200;
			MinFrac = 0.01f;

			Size = desc.velocityField->Size;
			DX = desc.cellWidth;
			DeltaTime = desc.deltaTime;
			VelocityField = desc.velocityField;
			LiquidSDF = desc.liquidSDF;
			WeightGrid = desc.weightGrid;

			PressureCells = std::vector<glm::ivec3>();
			for (int z = 1; z < Size.z - 1; z++) {
				for (int y = 1; y < Size.y - 1; y++) {
					for (int i = 1; i < Size.x - 1; i++) {
						if (LiquidSDF->Get(i, y, z) < 0) {
							PressureCells.push_back(glm::ivec3(i, y, z));
						}
					}
				}
			}

			MatSize = (int)PressureCells.size();
		}

		__device__ __host__ void InitGridIndexKeyMap() {
			KeyMap = GridIndexKeyMap(Size.x, Size.y, Size.z);
			for (unsigned int idx = 0; idx < PressureCells.size(); idx++) {
				KeyMap.insert(PressureCells[idx], idx);
			}
		}

		__device__ __host__ void HostFree() {

		}

		glm::vec3 Size;
		float DX;
		float Density;
		float DeltaTime;
		int MatSize;

		float PressureSolveTolerance;
		int maxCGIIterations;
		float MinFrac;

		std::vector<glm::ivec3> PressureCells;

		MACVelocityField* VelocityField;
		ParticleLevelSet* LiquidSDF;
		WeightGrid* WeightGrid;

		GridIndexKeyMap KeyMap;
	};
}

#endif // !PRESSURE_SOLVER_CUH
