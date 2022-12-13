#ifndef RIGID_BODY_IMPLEMENTATION_H
#define RIGID_BODY_IMPLEMENTATION_H

#include "Core/Structures/BoundingBox.h"

namespace vfd
{
	struct RigidBodyDescription;

	struct RigidBodyData
	{
		__host__ RigidBodyData() = default;
		__host__ RigidBodyData(const RigidBodyDescription& desc);

		glm::mat4x4 Transform;
		BoundingBox<glm::dvec3> Domain;

		glm::uvec3 Resolution;
		glm::dvec3 CellSize;
		glm::dvec3 CellSizeInverse;

		// -------- NODES --------
		// NodeElementCount
		// | A | B | C | D |
		// | A | B | C | D |
		// rows = NodeCount

		__host__ __device__ __forceinline__ double GetNode(unsigned int i, unsigned int j) const
		{
			return Nodes[i * NodeCount + j];
		}

		__host__ __device__ __forceinline__ double& GetNode(unsigned int i, unsigned int j)
		{
			return Nodes[i * NodeCount + j];
		}

		unsigned int NodeCount;
		unsigned int NodeElementCount;
		double* Nodes;

		// -------- CELL MAP --------
		// CellMapElementCount
		// | A | B | C | D |
		// | A | B | C | D |
		// rows = CellMapCount

		__host__ __device__ __forceinline__ unsigned  int GetCellMap(unsigned int i, unsigned int j) const
		{
			return CellMap[i * NodeCount + j];
		}

		__host__ __device__ __forceinline__ unsigned int& GetCellMap(unsigned int i, unsigned int j)
		{
			return CellMap[i * NodeCount + j];
		}

		unsigned int CellMapCount;
		unsigned int CellMapElementCount;
		unsigned int* CellMap;

		// -------- CELLS --------
		__host__ __device__ __forceinline__ unsigned  int GetCell(unsigned int i, unsigned int j, unsigned int k) const
		{
			return Cells[i + CellCount * (j + CellElementCount * k)];
		}

		__host__ __device__ __forceinline__ unsigned int& GetCell(unsigned int i, unsigned int j, unsigned int k)
		{
			return Cells[i + CellCount * (j + CellElementCount * k)];
		}

		size_t CellCount; // ie 2 
		unsigned int CellElementCount; // ie 2000
		unsigned int* Cells;
	};

	struct RigidBodyDeviceData
	{
		RigidBodyData* RigidBody;
		unsigned int* Cells;
		unsigned int* CellMap;
		double* Nodes;
	};
}

#endif // !RIGID_BODY_IMPLEMENTATION_H