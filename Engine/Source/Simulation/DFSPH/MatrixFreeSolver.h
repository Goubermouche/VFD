#ifndef MATRIX_FREE_SOLVER_H
#define MATRIX_FREE_SOLVER_H

#include "DFSPHSimulation.h"

namespace fe {
	
	class DFSPHSimulation;

	class Matrix3f8 {
	public:
		Scalar8 m[3][3];

		Matrix3f8() {  }

		inline void SetZero()
		{
			m[0][0].v = _mm256_setzero_ps();
			m[1][0].v = _mm256_setzero_ps();
			m[2][0].v = _mm256_setzero_ps();

			m[0][1].v = _mm256_setzero_ps();
			m[1][1].v = _mm256_setzero_ps();
			m[2][1].v = _mm256_setzero_ps();

			m[0][2].v = _mm256_setzero_ps();
			m[1][2].v = _mm256_setzero_ps();
			m[2][2].v = _mm256_setzero_ps();
		}

		inline Matrix3f8 operator * (const Scalar8& b) const
		{
			Matrix3f8 A;

			A.m[0][0] = m[0][0] * b;
			A.m[0][1] = m[0][1] * b;
			A.m[0][2] = m[0][2] * b;

			A.m[1][0] = m[1][0] * b;
			A.m[1][1] = m[1][1] * b;
			A.m[1][2] = m[1][2] * b;

			A.m[2][0] = m[2][0] * b;
			A.m[2][1] = m[2][1] * b;
			A.m[2][2] = m[2][2] * b;

			return A;
		}

		inline Matrix3f8& operator += (const Matrix3f8& a) {
			m[0][0] += a.m[0][0];
			m[1][0] += a.m[1][0];
			m[2][0] += a.m[2][0];

			m[0][1] += a.m[0][1];
			m[1][1] += a.m[1][1];
			m[2][1] += a.m[2][1];

			m[0][2] += a.m[0][2];
			m[1][2] += a.m[1][2];
			m[2][2] += a.m[2][2];
			return *this;
		}

		inline glm::mat3x3 Reduce() const {
			glm::mat3x3 A;
			A[0][0] = m[0][0].Reduce();
			A[0][1] = m[0][1].Reduce();
			A[0][2] = m[0][2].Reduce();

			A[1][0] = m[1][0].Reduce();
			A[1][1] = m[1][1].Reduce();
			A[1][2] = m[1][2].Reduce();

			A[2][0] = m[2][0].Reduce();
			A[2][1] = m[2][1].Reduce();
			A[2][2] = m[2][2].Reduce();
			return A;
		}
	};

	inline void DyadicProduct(const Scalar3f8& a, const Scalar3f8& b, Matrix3f8& res)
	{
		res.m[0][0] = _mm256_mul_ps(a.x().v, b.x().v); res.m[0][1] = _mm256_mul_ps(a.x().v, b.y().v); res.m[0][2] = _mm256_mul_ps(a.x().v, b.z().v);
		res.m[1][0] = _mm256_mul_ps(a.y().v, b.x().v); res.m[1][1] = _mm256_mul_ps(a.y().v, b.y().v); res.m[1][2] = _mm256_mul_ps(a.y().v, b.z().v);
		res.m[2][0] = _mm256_mul_ps(a.z().v, b.x().v); res.m[2][1] = _mm256_mul_ps(a.z().v, b.y().v); res.m[2][2] = _mm256_mul_ps(a.z().v, b.z().v);
	}

	class MatrixReplacement {
	public:
		typedef void(*MatrixVecProdFct) (const std::vector<float>&, std::vector<float>&, void*, DFSPHSimulation*);

		MatrixReplacement() = default;
		MatrixReplacement(const unsigned int dim, MatrixVecProdFct fct, void* userData, DFSPHSimulation* sim)
			: m_Dim(dim), m_MatrixVecProdFct(fct), m_UserData(userData), m_Base(sim)
		{}

		unsigned int m_Dim;
		void* m_UserData;
		/** matrix vector product callback */
		MatrixVecProdFct m_MatrixVecProdFct;
		DFSPHSimulation* m_Base;
	};

	class BlockJacobiPreconditioner3D {
	public:
		typedef void(*DiagonalMatrixElementFct) (const unsigned int, glm::mat3x3&, void*, DFSPHSimulation*);

		BlockJacobiPreconditioner3D() {}

		void Init(const unsigned int dim, DiagonalMatrixElementFct fct, void* userData, DFSPHSimulation* base) {
			m_dim = dim; 
			m_diagonalElementFct = fct;
			m_userData = userData;
			m_Base = base;
		}

		void Compute(const MatrixReplacement& mat) {
			m_invDiag.resize(m_dim);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static) 
				for (int i = 0; i < (int)m_dim; i++)
				{
					glm::mat3x3 res;
					m_diagonalElementFct(i, res, m_userData, m_Base);
					m_invDiag[i] = glm::inverse(res);
				}
			}
		}

		std::vector<float> Solve(const std::vector<float> b) {
			std::vector<float> x(b.size());

			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static) 
				for (int i = 0; i < (int)m_dim; i++)
				{
					glm::vec3 t = m_invDiag[i] * glm::vec3(b[3 * i + 0], b[3 * i + 1], b[3 * i + 2]);

					x[3 * i + 0] = t.x;
					x[3 * i + 1] = t.y;
					x[3 * i + 2] = t.z;
				}
			}

			return x;
		}
	public:
		unsigned int m_dim;
		DiagonalMatrixElementFct m_diagonalElementFct;
		void* m_userData;
		std::vector<glm::mat3x3> m_invDiag;
		DFSPHSimulation* m_Base;
	};

	static float SquaredNorm(const std::vector<float>& vec) {
		float val = 0.0f;
		for (size_t i = 0; i < vec.size(); i++)
		{
			val += vec[i] * vec[i];
		}
		return val;
	}

	static float Dot(const std::vector<float>& a, const std::vector<float>& b) {
		float val = 0.0f;
		for (size_t i = 0; i < a.size(); i++)
		{
			val += a[i] * b[i];
		}
		return val;
	}

	static void SetZero(std::vector<float>& vec) {
		for (size_t i = 0; i < vec.size(); i++)
		{
			vec[i] = 0;
		}
	}

	static std::vector<float> Subtract(const std::vector<float>& a, const std::vector<float>& b) {
		std::vector<float> res(a.size());
		for (size_t i = 0; i < a.size(); i++)
		{
			res[i] = a[i] - b[i];
		}
		return res;
	}

	static std::vector<float> Add(const std::vector<float>& a, const std::vector<float>& b) {
		std::vector<float> res(a.size());
		for (size_t i = 0; i < a.size(); i++)
		{
			res[i] = a[i] + b[i];
		}
		return res;
	}

	static std::vector<float> Multiply(const std::vector<float>& a, const std::vector<float>& b) {
		std::vector<float> res(a.size());
		for (size_t i = 0; i < a.size(); i++)
		{
			res[i] = a[i] * b[i];
		}
		return res;
	}

	static std::vector<float> Multiply(const std::vector<float>& a, float b) {
		std::vector<float> res(a.size());
		for (size_t i = 0; i < a.size(); i++)
		{
			res[i] = a[i] * b;
		}
		return res;
	}

	static std::vector<float> Multiply(const MatrixReplacement& lhs, const std::vector<float>& rhs) {
		std::vector<float> dst(rhs.size());

		lhs.m_MatrixVecProdFct(rhs, dst, lhs.m_UserData, lhs.m_Base);

		return dst;
	}

	static std::vector<float> Divide(const std::vector<float>& a, const std::vector<float>& b) {
		std::vector<float> res(a.size());
		for (size_t i = 0; i < a.size(); i++)
		{
			res[i] = a[i] / b[i];
		}
		return res;
	}

	class ConjugateFreeGradientSolver {
	public:
		ConjugateFreeGradientSolver() {
			Init();
		}

		void Init() {
			m_isInitialized = false;
			m_analysisIsOk = false;
			m_factorizationIsOk = false;
			m_MaxPressureSolverIterations = -1;
			m_tolerance = -1;
			m_PressureSolverIterations = 0;
		}

		BlockJacobiPreconditioner3D& GetPreconditioner() {
			return m_preconditioner;
		}

		const BlockJacobiPreconditioner3D& preconditioner() const { 
			return m_preconditioner; 
		}

		void Compute(const MatrixReplacement& A) {
			// grab(A.derived());
			m_matrixWrapper = A;
			m_preconditioner.Compute(m_matrixWrapper);
			m_isInitialized = true;
			m_analysisIsOk = true;
			m_factorizationIsOk = true;
		}

		void ConjugateGradient(/*matrixWrapper(mat)*/ const std::vector<float>& rhs, std::vector<float>& x
			/*preconditioner, iterations, error*/) {

			float tol = m_tolerance; // !
			int maxIters = m_MaxPressureSolverIterations;

			int n = m_matrixWrapper.m_Dim;

			std::vector<float> residual = Subtract(rhs, Multiply(m_matrixWrapper, x));

			float rhsNorm2 = SquaredNorm(rhs);
			if (rhsNorm2 == 0)
			{
				SetZero(x);
				m_PressureSolverIterations = 0;
				m_error = 0;
				return;
			}

			const float considerAsZero = (std::numeric_limits<float>::min)();
			float threshold = std::max(float(tol * tol * rhsNorm2), considerAsZero);
			float residualNorm2 = Dot(residual, residual);

			if (residualNorm2 < threshold)
			{
				m_PressureSolverIterations = 0;
				m_error = sqrt(residualNorm2 / rhsNorm2);
				return;
			}

			std::vector<float> p(n);
			p = m_preconditioner.Solve(residual);

			std::vector<float> z(n), tmp(n);
			float absNew = std::abs(Dot(residual, p));
			int i = 0;

			while (i < maxIters) {
				tmp = Multiply(m_matrixWrapper, p);

				float alpha = absNew / Dot(p, tmp);
				x = Add(x,  Multiply(p, alpha));
				residual = Subtract(residual, Multiply(tmp, alpha));

				residualNorm2 = SquaredNorm(residual);

				// std::cout << residualNorm2 << "   " << threshold << std::endl;
				if (residualNorm2 < threshold) {
					break;
				}

				z = m_preconditioner.Solve(residual);

				float absOld = absNew;
				absNew = std::abs(Dot(residual, z));
				float beta = absNew / absOld;
				p = Add(z, Multiply(p, beta));
				i++;
			}

			m_error = sqrt(residualNorm2 / rhsNorm2);
			m_PressureSolverIterations = i;
		}

		void SolveWithGuess(const std::vector<float>& b, const std::vector<float>& x0, std::vector<float>& xOut) {
			// derived = iterativeSolverBase
			// mat type = matrixReplacement
			//
			// derived()._solve_vector_with_guess_impl(b,dest.derived());

			xOut = x0;
			ConjugateGradient(b, xOut);
		}

		bool m_isInitialized;
		mutable bool m_analysisIsOk, m_factorizationIsOk;
		std::ptrdiff_t m_MaxPressureSolverIterations;
		float m_tolerance;
		float m_error;
		int m_PressureSolverIterations;

		BlockJacobiPreconditioner3D m_preconditioner;
		MatrixReplacement m_matrixWrapper;
	};
}

#endif // !MATRIX_FREE_SOLVER_H