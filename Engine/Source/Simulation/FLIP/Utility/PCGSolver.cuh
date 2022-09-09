#ifndef PCG_SOLVER_CUH
#define PCG_SOLVER_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/SparseMatrix.cuh"

namespace fe {
    template<class T>
    struct SparseColumnLowerFactor {

        unsigned int n;
        std::vector<T> invdiag;               // reciprocals of diagonal elements
        std::vector<T> value;                 // values below the diagonal, listed column by column
        std::vector<unsigned int> rowindex;   // a list of all row indices, for each column in turn
        std::vector<unsigned int> colstart;   // where each column begins in rowindex (plus an extra entry at the end, of #nonzeros)
        std::vector<T> adiag;                 // just used in factorization: minimum "safe" diagonal entry allowed

        explicit SparseColumnLowerFactor(unsigned int size = 0)
            : n(size), invdiag(size), colstart(size + 1), adiag(size) {}

        void clear(void) {
            n = 0;
            invdiag.clear();
            value.clear();
            rowindex.clear();
            colstart.clear();
            adiag.clear();
        }

        void resize(unsigned int size) {
            n = size;
            invdiag.resize(n);
            colstart.resize(n + 1);
            adiag.resize(n);
        }
    };

    inline double dot(const std::vector<double>& x, const std::vector<double>& y) {
        //return cblas_ddot((int)x.size(), &x[0], 1, &y[0], 1); 

        double sum = 0;
        for (size_t i = 0; i < x.size(); i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    // inf-norm (maximum absolute value: index of max returned) ==================

    inline int indexAbsMax(const std::vector<double>& x) {
        //return cblas_idamax((int)x.size(), &x[0], 1); 

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

    // inf-norm (maximum absolute value) =========================================
    // technically not part of BLAS, but useful

    inline double absMax(const std::vector<double>& x) {
        return std::fabs(x[indexAbsMax(x)]);
    }

    // saxpy (y=alpha*x+y) =======================================================

    inline void addScaled(double alpha, const std::vector<double>& x, std::vector<double>& y) {
        //cblas_daxpy((int)x.size(), alpha, &x[0], 1, &y[0], 1); 

        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    template<class T>
    void solveLower(const SparseColumnLowerFactor<T>& factor, const std::vector<T>& rhs,
        std::vector<T>& result) {

        result = rhs;
        for (unsigned int i = 0; i < factor.n; i++) {
            result[i] *= factor.invdiag[i];
            for (unsigned int j = factor.colstart[i]; j < factor.colstart[i + 1]; j++) {
                result[factor.rowindex[j]] -= factor.value[j] * result[i];
            }
        }
    }


    // solve L^T*result=rhs
    template<class T>
    void solveLowerTransposeInPlace(const SparseColumnLowerFactor<T>& factor, std::vector<T>& x)
    {
        unsigned int i = factor.n;
        do {
            i--;
            for (unsigned int j = factor.colstart[i]; j < factor.colstart[i + 1]; j++) {
                x[i] -= factor.value[j] * x[factor.rowindex[j]];
            }
            x[i] *= factor.invdiag[i];
        } while (i != 0);
    }

    template<class T>
    void factorModifiedIncompleteColesky0(const SparseMatrix<T>& matrix,
        SparseColumnLowerFactor<T>& factor,
        T modificationParameter = 0.97,
        T minDiagonalRatio = 0.25) {

        // first copy lower triangle of matrix into factor (Note: assuming A is symmetric of course!)
        factor.resize(matrix.n);
        std::fill(factor.invdiag.begin(), factor.invdiag.end(), 0); // important: eliminate old values from previous solves!
        factor.value.resize(0);
        factor.rowindex.resize(0);
        std::fill(factor.adiag.begin(), factor.adiag.end(), 0);

        for (unsigned int i = 0; i < matrix.n; i++) {
            factor.colstart[i] = (unsigned int)factor.rowindex.size();
            for (size_t j = 0; j < matrix.index[i].size(); j++) {
                if (matrix.index[i][j] > i) {
                    factor.rowindex.push_back(matrix.index[i][j]);
                    factor.value.push_back(matrix.value[i][j]);
                }
                else if (matrix.index[i][j] == i) {
                    factor.invdiag[i] = factor.adiag[i] = matrix.value[i][j];
                }
            }
        }
        factor.colstart[matrix.n] = (unsigned int)factor.rowindex.size();
        // now do the incomplete factorization (figure out numerical values)

        // MATLAB code:
        // L=tril(A);
        // for k=1:size(L,2)
        //   L(k,k)=sqrt(L(k,k));
        //   L(k+1:end,k)=L(k+1:end,k)/L(k,k);
        //   for j=find(L(:,k))'
        //     if j>k
        //       fullupdate=L(:,k)*L(j,k);
        //       incompleteupdate=fullupdate.*(A(:,j)~=0);
        //       missing=sum(fullupdate-incompleteupdate);
        //       L(j:end,j)=L(j:end,j)-incompleteupdate(j:end);
        //       L(j,j)=L(j,j)-omega*missing;
        //     end
        //   end
        // end

        for (unsigned int k = 0; k < matrix.n; k++) {
            if (factor.adiag[k] == 0) {
                // null row/column
                continue;
            }

            // figure out the final L(k,k) entry
            if (factor.invdiag[k] < minDiagonalRatio * factor.adiag[k]) {
                // drop to Gauss-Seidel here if the pivot looks dangerously small
                factor.invdiag[k] = 1 / sqrt(factor.adiag[k]);
            }
            else {
                factor.invdiag[k] = 1 / sqrt(factor.invdiag[k]);
            }

            // finalize the k'th column L(:,k)
            for (unsigned int p = factor.colstart[k]; p < factor.colstart[k + 1]; p++) {
                factor.value[p] *= factor.invdiag[k];
            }

            // incompletely eliminate L(:,k) from future columns, modifying diagonals
            for (unsigned int p = factor.colstart[k]; p < factor.colstart[k + 1]; p++) {
                unsigned int j = factor.rowindex[p]; // work on column j
                T multiplier = factor.value[p];
                T missing = 0;
                unsigned int a = factor.colstart[k];
                // first look for contributions to missing from dropped entries above the diagonal in column j
                unsigned int b = 0;
                while (a < factor.colstart[k + 1] && factor.rowindex[a] < j) {
                    // look for factor.rowindex[a] in matrix.index[j] starting at b
                    while (b < matrix.index[j].size()) {
                        if (matrix.index[j][b] < factor.rowindex[a]) {
                            b++;
                        }
                        else if (matrix.index[j][b] == factor.rowindex[a]) {
                            break;
                        }
                        else {
                            missing += factor.value[a];
                            break;
                        }
                    }
                    a++;
                }

                // adjust the diagonal j,j entry
                if (a < factor.colstart[k + 1] && factor.rowindex[a] == j) {
                    factor.invdiag[j] -= multiplier * factor.value[a];
                }
                a++;

                // and now eliminate from the nonzero entries below the diagonal in column j (or add to missing if we can't)
                b = factor.colstart[j];
                while (a < factor.colstart[k + 1] && b < factor.colstart[j + 1]) {
                    if (factor.rowindex[b] < factor.rowindex[a]) {
                        b++;
                    }
                    else if (factor.rowindex[b] == factor.rowindex[a]) {
                        factor.value[b] -= multiplier * factor.value[a];
                        a++;
                        b++;
                    }
                    else {
                        missing += factor.value[a];
                        a++;
                    }
                }

                // and if there's anything left to do, add it to missing
                while (a < factor.colstart[k + 1]) {
                    missing += factor.value[a];
                    a++;
                }

                // and do the final diagonal adjustment from the missing entries
                factor.invdiag[j] -= modificationParameter * multiplier * missing;
            }
        }
    }

    template <class T>
    struct PCGSolver {

        PCGSolver() {
            setSolverParameters(1e-12, 100, 0.97, 0.25);
        }

        void setSolverParameters(T tolerance,
            int maxiter,
            T MICParameter = 0.97,
            T diagRatio = 0.25) {

            toleranceFactor = tolerance;
            if (toleranceFactor < 1e-30) {
                toleranceFactor = 1e-30;
            }
            maxIterations = maxiter;
            modifiedIncompleteCholeskyParameter = MICParameter;
            minDiagonalRatio = diagRatio;
        }

        bool solve(const SparseMatrix<T>& matrix, const std::vector<T>& rhs,
            std::vector<T>& result, T& residualOut, int& iterationsOut) {

            unsigned int n = matrix.n;
            if (m.size() != n) {
                m.resize(n);
                s.resize(n);
                z.resize(n);
                r.resize(n);
            }
            std::fill(result.begin(), result.end(), 0);

            r = rhs;
            residualOut = absMax(r);
            if (residualOut == 0) {
                iterationsOut = 0;
                return true;
            }
            double tol = toleranceFactor * residualOut;

            formPreconditioner(matrix);
            applyPreconditioner(r, z);
            double rho = dot(z, r);
            if (rho == 0 || rho != rho) {
                iterationsOut = 0;
                return false;
            }

            s = z;
            fixedMatrix.fromMatrix(matrix);

            int iteration;
            for (iteration = 0; iteration < maxIterations; iteration++) {
                multiply(fixedMatrix, s, z);
                double alpha = rho / dot(s, z);
                addScaled(alpha, s, result);
                addScaled(-alpha, z, r);

                residualOut = absMax(r);
                if (residualOut <= tol) {
                    iterationsOut = iteration + 1;
                    return true;
                }

                applyPreconditioner(r, z);
                double rhoNew = dot(z, r);
                double beta = rhoNew / rho;
                addScaled(beta, s, z);
                s.swap(z); // s=beta*s+z
                rho = rhoNew;
            }

            iterationsOut = iteration;
            return false;
        }

    protected:

        // internal structures
        SparseColumnLowerFactor<T> icfactor; // modified incomplete cholesky factor
        std::vector<T> m, z, s, r; // temporary vectors for PCG
        FixedSparseMatrix<T> fixedMatrix; // used within loop

        // parameters
        T toleranceFactor;
        int maxIterations;
        T modifiedIncompleteCholeskyParameter;
        T minDiagonalRatio;

        void formPreconditioner(const SparseMatrix<T>& matrix) {
            factorModifiedIncompleteColesky0(matrix, icfactor);
        }

        void applyPreconditioner(const std::vector<T>& x, std::vector<T>& result) {
            solveLower(icfactor, x, result);
            solveLowerTransposeInPlace(icfactor, result);
        }

    };
}

#endif // !PCG_SOLVER_CUH
