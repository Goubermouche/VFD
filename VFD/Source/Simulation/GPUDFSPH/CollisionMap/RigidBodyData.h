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

		__host__ __device__ __forceinline__ double GetNode(unsigned int i, unsigned int j) const
		{
			return Nodes[i * NodeElementCount + j];
		}

		__host__ __device__ __forceinline__ double& GetNode(unsigned int i, unsigned int j)
		{
			return Nodes[i * NodeElementCount + j];
		}

		__host__ __device__ __forceinline__ unsigned  int GetCellMap(unsigned int i, unsigned int j) const
		{
			return CellMap[i * CellMapElementCount + j];
		}

		__host__ __device__ __forceinline__ unsigned int& GetCellMap(unsigned int i, unsigned int j)
		{
			return CellMap[i * CellMapElementCount + j];
		}

		__host__ __device__ __forceinline__ unsigned  int GetCell(unsigned int i, unsigned int j, unsigned int k) const
		{
			return Cells[i * CellElementCount * 32 + (j * 32 + k)];
		}

		__host__ __device__ __forceinline__ unsigned int& GetCell(unsigned int i, unsigned int j, unsigned int k)
		{
			return Cells[i * CellElementCount * 32 + (j * 32 + k)];
		}

		__host__ __device__ __forceinline__ glm::vec3& GetBoundaryXJ(unsigned int i)
		{
			return BoundaryXJ[i];
		}

		__host__ __device__ __forceinline__ float& GetBoundaryVolume(unsigned int i)
		{
			return BoundaryVolume[i];
		}

		glm::mat4x4 Transform;
		BoundingBox<glm::dvec3> Domain;

		glm::uvec3 Resolution;
		glm::dvec3 CellSize;
		glm::dvec3 CellSizeInverse;

		// Nodes
		unsigned int NodeCount;
		unsigned int NodeElementCount;
		double* Nodes;

		// Cell map
		unsigned int CellMapCount;
		unsigned int CellMapElementCount;
		unsigned int* CellMap;

		// Cells
		size_t CellCount;
		unsigned int CellElementCount;
		unsigned int* Cells;

		glm::vec3* BoundaryXJ;
		float* BoundaryVolume;
	};

	struct RigidBodyDeviceData
	{
		RigidBodyData* RigidBody;
		unsigned int* Cells;
		unsigned int* CellMap;
		double* Nodes;
		glm::vec3* BoundaryXJ;
		float* BoundaryVolume;
	};
}

#endif // !RIGID_BODY_IMPLEMENTATION_H