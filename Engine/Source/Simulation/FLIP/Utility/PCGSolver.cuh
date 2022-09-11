#ifndef PCG_SOLVER_CUH
#define PCG_SOLVER_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/SparseMatrix.cuh"

namespace fe {
    template<class T>
    struct SparseColumnLowerFactor {
        unsigned int Count;
        std::vector<T> InvDiagonal;            
        std::vector<T> Value;               
        std::vector<unsigned int> RowIndices;   
        std::vector<unsigned int> ColumnStarts; 
        std::vector<T> ADiagonal; 

        explicit SparseColumnLowerFactor(unsigned int size = 0)
            : Count(size), InvDiagonal(size), ColumnStarts(size + 1), ADiagonal(size)
        {}

        void Clear(void) {
            Count = 0;
            InvDiagonal.clear();
            Value.clear();
            RowIndices.clear();
            ColumnStarts.clear();
            ADiagonal.clear();
        }

        void Resize(unsigned int size) {
            Count = size;
            InvDiagonal.resize(Count);
            ColumnStarts.resize(Count + 1);
            ADiagonal.resize(Count);
        }
    };

    inline double Dot(const std::vector<double>& x, const std::vector<double>& y) {
        double sum = 0;
        for (size_t i = 0; i < x.size(); i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    inline int IndexAbsMax(const std::vector<double>& x) {
        int maxind = 0;
        double maxvalue = 0;
        for (size_t i = 0; i < x.size(); i++) {
            if (fabs(x[i]) > maxvalue) {
                maxvalue = fabs(x[i]);
                maxind = i;
            }
        }
        return maxind;
    }

    inline double AbsMax(const std::vector<double>& x) {
        return std::fabs(x[IndexAbsMax(x)]);
    }

    inline void AddScaled(double alpha, const std::vector<double>& x, std::vector<double>& y) {
        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    template<class T>
    void SolveLower(const SparseColumnLowerFactor<T>& factor, const std::vector<T>& rhs, std::vector<T>& result) {
        result = rhs;
        for (unsigned int i = 0; i < factor.Count; i++) {
            result[i] *= factor.InvDiagonal[i];
            for (unsigned int j = factor.ColumnStarts[i]; j < factor.ColumnStarts[i + 1]; j++) {
                result[factor.RowIndices[j]] -= factor.Value[j] * result[i];
            }
        }
    }

    template<class T>
    void SolveLowerTransposeInPlace(const SparseColumnLowerFactor<T>& factor, std::vector<T>& x)
    {
        unsigned int i = factor.Count;
        do {
            i--;
            for (unsigned int j = factor.ColumnStarts[i]; j < factor.ColumnStarts[i + 1]; j++) {
                x[i] -= factor.Value[j] * x[factor.RowIndices[j]];
            }
            x[i] *= factor.InvDiagonal[i];
        } while (i != 0);
    }

    template<class T>
    void FactorModifiedIncompleteCholesky(const SparseMatrix<T>& matrix, SparseColumnLowerFactor<T>& factor, T modificationParameter = 0.97, T minDiagonalRatio = 0.25) {
        factor.Resize(matrix.Count);
        std::fill(factor.InvDiagonal.begin(), factor.InvDiagonal.end(), 0); // important: eliminate old values from previous solves!
        factor.Value.resize(0);
        factor.RowIndices.resize(0);
        std::fill(factor.ADiagonal.begin(), factor.ADiagonal.end(), 0);

        for (unsigned int i = 0; i < matrix.Count; i++) {
            factor.ColumnStarts[i] = (unsigned int)factor.RowIndices.size();
            for (size_t j = 0; j < matrix.Indices[i].size(); j++) {
                if (matrix.Indices[i][j] > i) {
                    factor.RowIndices.push_back(matrix.Indices[i][j]);
                    factor.Value.push_back(matrix.Value[i][j]);
                }
                else if (matrix.Indices[i][j] == i) {
                    factor.InvDiagonal[i] = factor.ADiagonal[i] = matrix.Value[i][j];
                }
            }
        }
        factor.ColumnStarts[matrix.Count] = (unsigned int)factor.RowIndices.size();

        for (unsigned int k = 0; k < matrix.Count; k++) {
            if (factor.ADiagonal[k] == 0) {
                continue;
            }

            if (factor.InvDiagonal[k] < minDiagonalRatio * factor.ADiagonal[k]) {
                factor.InvDiagonal[k] = 1 / sqrt(factor.ADiagonal[k]);
            }
            else {
                factor.InvDiagonal[k] = 1 / sqrt(factor.InvDiagonal[k]);
            }

            for (unsigned int p = factor.ColumnStarts[k]; p < factor.ColumnStarts[k + 1]; p++) {
                factor.Value[p] *= factor.InvDiagonal[k];
            }

            for (unsigned int p = factor.ColumnStarts[k]; p < factor.ColumnStarts[k + 1]; p++) {
                unsigned int j = factor.RowIndices[p];
                T multiplier = factor.Value[p];
                T missing = 0;
                unsigned int a = factor.ColumnStarts[k];
                unsigned int b = 0;
                while (a < factor.ColumnStarts[k + 1] && factor.RowIndices[a] < j) {
                    while (b < matrix.Indices[j].size()) {
                        if (matrix.Indices[j][b] < factor.RowIndices[a]) {
                            b++;
                        }
                        else if (matrix.Indices[j][b] == factor.RowIndices[a]) {
                            break;
                        }
                        else {
                            missing += factor.Value[a];
                            break;
                        }
                    }
                    a++;
                }

                if (a < factor.ColumnStarts[k + 1] && factor.RowIndices[a] == j) {
                    factor.InvDiagonal[j] -= multiplier * factor.Value[a];
                }
                a++;

                b = factor.ColumnStarts[j];
                while (a < factor.ColumnStarts[k + 1] && b < factor.ColumnStarts[j + 1]) {
                    if (factor.RowIndices[b] < factor.RowIndices[a]) {
                        b++;
                    }
                    else if (factor.RowIndices[b] == factor.RowIndices[a]) {
                        factor.Value[b] -= multiplier * factor.Value[a];
                        a++;
                        b++;
                    }
                    else {
                        missing += factor.Value[a];
                        a++;
                    }
                }

                while (a < factor.ColumnStarts[k + 1]) {
                    missing += factor.Value[a];
                    a++;
                }

                factor.InvDiagonal[j] -= modificationParameter * multiplier * missing;
            }
        }
    }

    template <class T>
    struct PCGSolver {
        PCGSolver() {
            SetSolverParameters(1e-12, 100, 0.97, 0.25);
        }

        void SetSolverParameters(T tolerance, int maxiter, T MICParameter = 0.97, T diagRatio = 0.25) {

            ToleranceFactor = tolerance;
            if (ToleranceFactor < 1e-30) {
                ToleranceFactor = 1e-30;
            }
            MaxIterations = maxiter;
            ModifiedIncompleteCholeskyParameter = MICParameter;
            MinDiagonalRatio = diagRatio;
        }

        bool Solve(const SparseMatrix<T>& matrix, const std::vector<T>& rhs, std::vector<T>& result, T& residualOut, int& iterationsOut) {
            unsigned int Count = matrix.Count;
            if (M.size() != Count) {
                M.resize(Count);
                S.resize(Count);
                Z.resize(Count);
                R.resize(Count);
            }
            std::fill(result.begin(), result.end(), 0);

            R = rhs;
            residualOut = AbsMax(R);
            if (residualOut == 0) {
                iterationsOut = 0;
                return true;
            }
            double tol = ToleranceFactor * residualOut;

            FormPreconditioner(matrix);
            ApplyPreconditioner(R, Z);
            double rho = Dot(Z, R);
            if (rho == 0 || rho != rho) {
                iterationsOut = 0;
                return false;
            }

            S = Z;
            FixedMatrix.FromMatrix(matrix);

            int iteration;
            {
                TIME_SCOPE("Solve...")
                for (iteration = 0; iteration < MaxIterations; iteration++) {
                    Multiply(FixedMatrix, S, Z);
                    double alpha = rho / Dot(S, Z);
                    AddScaled(alpha, S, result);
                    AddScaled(-alpha, Z, R);

                    residualOut = AbsMax(R);
                    if (residualOut <= tol) {
                        iterationsOut = iteration + 1;
                        return true;
                    }

                    ApplyPreconditioner(R, Z);
                    double rhoNew = Dot(Z, R);
                    double beta = rhoNew / rho;
                    AddScaled(beta, S, Z);
                    S.swap(Z);
                    rho = rhoNew;
                }
            }

            iterationsOut = iteration;
            return false;
        }
    protected:
        void FormPreconditioner(const SparseMatrix<T>& matrix) {
            FactorModifiedIncompleteCholesky(matrix, IncompleteCholensky);
        }

        void ApplyPreconditioner(const std::vector<T>& x, std::vector<T>& result) {
            SolveLower(IncompleteCholensky, x, result);
            SolveLowerTransposeInPlace(IncompleteCholensky, result);
        }
    protected:
        SparseColumnLowerFactor<T> IncompleteCholensky;
        std::vector<T> M;
        std::vector<T> Z;
        std::vector<T> S;
        std::vector<T> R;
        FixedSparseMatrix<T> FixedMatrix;

        T ToleranceFactor;
        int MaxIterations;
        T ModifiedIncompleteCholeskyParameter;
        T MinDiagonalRatio;
    };
}

#endif // !PCG_SOLVER_CUH
