#ifndef MATRIX_FREE_SOLVER_H
#define MATRIX_FREE_SOLVER_H

#include "DFSPHSimulation.h"

namespace fe {
	class DFSPHSimulation;

	class MatrixReplacement {
	public:
		typedef void(*MatrixVecProdFct) (const float*, float*, void*, DFSPHSimulation*);

		MatrixReplacement(const unsigned int dim, MatrixVecProdFct fct, void* userData, DFSPHSimulation* sim)
			: m_Dim(dim), m_MatrixVecProdFct(fct), m_UserData(userData) 
		{}

	private:
		unsigned int m_Dim;
		void* m_UserData;
		/** matrix vector product callback */
		MatrixVecProdFct m_MatrixVecProdFct;
	};
}

#endif // !MATRIX_FREE_SOLVER_H