#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH

#include "pch.h"

namespace fe {
    template<class T>
    struct SparseMatrix {
        unsigned int Count;                                
        std::vector<std::vector<unsigned int> > Indices;   
        std::vector<std::vector<T> > Value;                

        SparseMatrix(unsigned int size = 0, unsigned int expectedNonZeros = 7) 
            : Count(size), Indices(size), Value(size) {
            for (unsigned int i = 0; i < Count; i++) {
                Indices[i].reserve(expectedNonZeros);
                Value[i].reserve(expectedNonZeros);
            }
        }

        void Clear(void) {
            Count = 0;
            Indices.clear();
            Value.clear();
        }

        void Zero(void) {
            for (unsigned int i = 0; i < Count; i++) {
                Indices[i].resize(0);
                Value[i].resize(0);
            }
        }

        void Resize(int size) {
            Count = size;
            Indices.resize(size);
            Value.resize(size);
        }

        T operator()(int i, int j) const {
            for (size_t k = 0; k < Indices[i].size(); k++) {
                if (Indices[i][k] == j) {
                    return Value[i][k];
                }
                else if (Indices[i][k] > j) {
                    return 0;
                }
            }
            return 0;
        }

        void Set(int i, int j, T newValue) {
            if (i == -1 || j == -1) {
                return;
            }

            for (size_t k = 0; k < Indices[i].size(); k++) {
                if (Indices[i][k] == (unsigned int)j) {
                    Value[i][k] = newValue;
                    return;
                }
                else if (Indices[i][k] > (unsigned int)j) {
                    Indices[i].insert(Indices[i].begin() + k, j);
                    Value[i].insert(Value[i].begin() + k, newValue);
                    return;
                }
            }
            Indices[i].push_back(j);
            Value[i].push_back(newValue);
        }

        void Add(int i, int j, T inc)
        {
            if (i == -1 || j == -1) {
                return;
            }

            for (size_t k = 0; k < Indices[i].size(); k++) {
                if (Indices[i][k] == (unsigned int)j) {
                    Value[i][k] += inc;
                    return;
                }
                else if (Indices[i][k] > (unsigned int)j) {
                    Indices[i].insert(Indices[i].begin() + k, j);
                    Value[i].insert(Value[i].begin() + k, inc);
                    return;
                }
            }
            Indices[i].push_back(j);
            Value[i].push_back(inc);
        }
    };

    template<class T>
    struct FixedSparseMatrix {
        unsigned int Count;                      
        std::vector<T> Value;
        std::vector<unsigned int> ColumnIndeces;
        std::vector<unsigned int> RowStart;

        explicit FixedSparseMatrix(unsigned int size = 0)
            : Count(size), Value(0), ColumnIndeces(0), RowStart(size + 1)
        {}

        void Clear(void)
        {
            Count = 0;
            Value.clear();
            ColumnIndeces.clear();
            RowStart.clear();
        }

        void Resize(int size)
        {
            Count = size;
            RowStart.resize(Count + 1);
        }

        void FromMatrix(const SparseMatrix<T>& matrix)
        {
            Resize(matrix.Count);
            RowStart[0] = 0;
            for (unsigned int i = 0; i < Count; i++) {
                RowStart[i + 1] = RowStart[i] + matrix.Indices[i].size();
            }

            Value.resize(RowStart[Count]);
            ColumnIndeces.resize(RowStart[Count]);

            size_t j = 0;
            for (size_t i = 0; i < Count; i++) {
                for (size_t k = 0; k < matrix.Indices[i].size(); k++) {
                    Value[j] = matrix.Value[i][k];
                    ColumnIndeces[j] = matrix.Indices[i][k];
                    j++;
                }
            }
        }
    };

    template<class T>
    void Multiply(const FixedSparseMatrix<T>& matrix, const std::vector<T>& x, std::vector<T>& result) {
        result.resize(matrix.Count);
        for (size_t i = 0; i < matrix.Count; i++) {
            result[i] = 0;
            for (size_t j = matrix.RowStart[i]; j < matrix.RowStart[i + 1]; j++) {
                result[i] += matrix.Value[j] * x[matrix.ColumnIndeces[j]];
            }
        }
    }
}

#endif // !SPARSE_MATRIX_CUH
