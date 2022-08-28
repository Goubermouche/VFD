#include "MeshLevelSet.cuh"
#include "pch.h"

namespace fe {
	__device__ Array3D<float> d_SDFPhi;
	__device__ Array3D<int> d_SDFClosestTriangles;
	__device__ Array3D<float> symbol;

	static __global__ void TestKernel() {
		printf("Test Kernel\n");
	}

	template <typename T>
	__device__ T NewClamp(const T& n, const T& lower, const T& upper) {
		return max(lower, min(n, upper));
	}

	__device__ glm::vec3 NewGridIndexToPosition(int i, int j, int k, float dx) {
		return { i * dx, j * dx, k * dx };
	}

	__device__ float NewPointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2) {
		glm::vec3 dx = x2 - x1;
		float m2 = glm::length2(dx);
		float s12 = glm::dot(x2 - x0, dx) / m2;
		if (s12 < 0) {
			s12 = 0;
		}
		else if (s12 > 1) {
			s12 = 1;
		}

		return glm::length(x0 - (s12 * x1 + (+-s12) * x2));
	}

	__device__ float NewPointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3) {
		glm::vec3 x13 = x1 - x3;
		glm::vec3 x23 = x2 - x3;
		glm::vec3 x03 = x0 - x3;

		float m13 = glm::length2(x13);
		float m23 = glm::length2(x23);
		float d = glm::dot(x13, x23);
		float invdet = 1.0f / fmax(m13 * m23 - d * d, 1e-30f);
		float a = glm::dot(x13, x03);
		float b = glm::dot(x23, x03);

		float w23 = invdet * (m23 * a - d * b);
		float w31 = invdet * (m13 * b - d * a);
		float w12 = 1 - w23 - w31;

		if (w23 >= 0 && w31 >= 0 && w12 >= 0) {
			return glm::length(x0 - (w23 * x1 + w31 * x2 + w12 * x3));
		}
		else {
			if (w23 > 0) {
				float d1 = NewPointToSegmentDistance(x0, x1, x2);
				float d2 = NewPointToSegmentDistance(x0, x1, x3);
				return fmin(d1, d2);
			}
			else if (w31 > 0) {
				// this rules out edge 1-3
				float d1 = NewPointToSegmentDistance(x0, x1, x2);
				float d2 = NewPointToSegmentDistance(x0, x2, x3);
				return fmin(d1, d2);
			}
			else {
				// w12 must be >0, ruling out edge 1-2
				float d1 = NewPointToSegmentDistance(x0, x1, x3);
				float d2 = NewPointToSegmentDistance(x0, x2, x3);
				return fmin(d1, d2);
			}
		}
	}

	static __global__ void ComputeExactBandDistanceFieldKernel(int bandWidth, float DX, glm::ivec3 size, Array3D<int>& intersectionCounts, const glm::vec3* vertices, int vertexCount,const glm::ivec3* triangles, int triangleCount) {
		const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		float invDX = 1.0f / DX;

		glm::ivec3 t = triangles[index];
		glm::vec3 p = vertices[t.x];
		glm::vec3 q = vertices[t.y];
		glm::vec3 r = vertices[t.z];

		float fip = p.x * invDX;
		float fjp = p.y * invDX;
		float fkp = p.z * invDX;

		float fiq = q.x * invDX;
		float fjq = q.y * invDX;
		float fkq = q.z * invDX;

		float fir = r.x * invDX;
		float fjr = r.y * invDX;
		float fkr = r.z * invDX;

		int i0 = NewClamp(int(fmin(fip, fmin(fiq, fir))) - bandWidth, 0, size.x - 1);
		int j0 = NewClamp(int(fmin(fjp, fmin(fjq, fjr))) - bandWidth, 0, size.y - 1);
		int k0 = NewClamp(int(fmin(fkp, fmin(fkq, fkr))) - bandWidth, 0, size.z - 1);

		int i1 = NewClamp(int(fmax(fip, fmax(fiq, fir))) + bandWidth + 1, 0, size.x - 1);
		int j1 = NewClamp(int(fmax(fjp, fmax(fjq, fjr))) + bandWidth + 1, 0, size.y - 1);
		int k1 = NewClamp(int(fmax(fkp, fmax(fkq, fkr))) + bandWidth + 1, 0, size.z - 1);

		for (int k = k0; k <= k1; k++) {
			for (int j = j0; j <= j1; j++) {
				for (int i = i0; i <= i1; i++) {
					glm::vec3 gPos = NewGridIndexToPosition(i, j, k, DX);
					float d = NewPointToTriangleDistance(gPos, p, q, r);
					if (d < d_SDFPhi(i, j, k)) {
						d_SDFPhi.Set(i, j, k, d);
						d_SDFClosestTriangles.Set(i, j, k, index);
					}
				}
			}
		}
	}

	__host__ void MeshLevelSet::CalculateSDFNew(const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount, int bandWidth)
	{
		//MeshVertices = vertices;
		//MeshVertexCount = vertexCount;
		//MeshTriangles = triangles;
		//MeshTriangleCount = triangleCount;

		//Array3D<int> intersectionCounts;
	    // intersectionCounts.Init(Phi.Size.x, Phi.Size.y, Phi.Size.z);

		//glm::vec3 size = Phi.Size;

		//Phi.Fill((size.x + size.y + size.z) * DX);
		//ClosestTriangles.Fill(-1);
		//intersectionCounts.Fill(0);

		//// Init memory
		//Array3D<float> PhiDEVICE = Phi.UploadToDevice();
		//Array3D<int> ClosestTrianglesDEVICE = ClosestTriangles.UploadToDevice();
		//intersectionCounts = intersectionCounts.UploadToDevice();

		//COMPUTE_SAFE(cudaMemcpyToSymbol(d_SDFPhi, &PhiDEVICE, sizeof(Array3D<float>)));
		//COMPUTE_SAFE(cudaMemcpyToSymbol(d_SDFClosestTriangles, &ClosestTrianglesDEVICE, sizeof(Array3D<int>)));

		//COMPUTE_SAFE(cudaMalloc((void**)&MeshVertices, sizeof(float) * 3 * vertexCount));
		//COMPUTE_SAFE(cudaMalloc((void**)&MeshTriangles, sizeof(int) * 3 * triangleCount));

		//// Initialize distances near the mesh
		//{
		//	int threadCount;
		//	int blockCount;
		//	ComputeGridSize(MeshTriangleCount, 256, blockCount, threadCount);
		//	ComputeExactBandDistanceFieldKernel <<< threadCount, blockCount >>> (bandWidth, DX, size, intersectionCounts, MeshVertices, MeshVertexCount, MeshTriangles, MeshTriangleCount);
		//	COMPUTE_SAFE(cudaDeviceSynchronize());
		//}

		//// Free memory
		//Phi = PhiDEVICE.UploadToHost();
		//ClosestTriangles = ClosestTrianglesDEVICE.UploadToHost();
		//intersectionCounts.Free();

		//COMPUTE_SAFE(cudaMemcpyFromSymbol(&Phi, d_SDFPhi, sizeof(Phi), 0, cudaMemcpyDeviceToHost));
		//COMPUTE_SAFE(cudaMemcpyFromSymbol(&ClosestTriangles, d_SDFClosestTriangles, sizeof(Array3D<int>)));

		//COMPUTE_SAFE(cudaFree((void**)MeshVertices));
		//COMPUTE_SAFE(cudaFree((void**)MeshTriangles));

		//ERR(Phi.Get(0));

		glm::vec3 size = Phi.Size;
		Phi.Fill((size.x + size.y + size.z) * DX);

		//auto PhiDevice = Phi.UploadToDevice();
		//COMPUTE_SAFE(cudaMemcpyToSymbol(d_SDFPhi, &PhiDevice, sizeof(Array3D<float>)));

		//Phi = PhiDevice.UploadToHost();
		//COMPUTE_SAFE(cudaMemcpyFromSymbol(&Phi, d_SDFPhi, sizeof(Phi), 0, cudaMemcpyDeviceToHost));

		// __________________________
		/*Array3D<float>* device;
		Array3D<float>* host;
		device = new Array3D<float>();
		host = new Array3D<float>();

		host->Init(10, 10, 10);
		host->Fill(5.0f);

		COMPUTE_SAFE(cudaMalloc(&device, sizeof(Array3D<float>)));
		COMPUTE_SAFE(cudaMemcpy(device, host, sizeof(Array3D<float>), cudaMemcpyHostToDevice));

		float* host_grid;
		COMPUTE_SAFE(cudaMalloc(&host_grid, sizeof(float)));
		COMPUTE_SAFE(cudaMemcpy(host_grid, host->Grid, sizeof(float), cudaMemcpyHostToDevice));
		COMPUTE_SAFE(cudaMemcpy(&(device->Grid), &host_grid, sizeof(void*), cudaMemcpyHostToDevice));

		COMPUTE_SAFE(cudaMemcpy(host->Grid, host_grid, sizeof(float), cudaMemcpyDeviceToHost));
		ERR(host->Grid[0]);*/
		// __________________________

		Array3D<float> host;
		host.Init(10, 10, 10);
		host.Fill(5);
		Array3D<float> device = host;

		COMPUTE_SAFE(cudaMalloc((void**)&device.Grid, host.GetSize()));
		COMPUTE_SAFE(cudaMemcpy(device.Grid, host.Grid, host.GetSize(), cudaMemcpyHostToDevice));
		COMPUTE_SAFE(cudaMemcpyToSymbol(symbol, &device, sizeof(device)));

		COMPUTE_SAFE(cudaMemcpyFromSymbol(&host, symbol, sizeof(void*), 0, cudaMemcpyDeviceToHost));
		COMPUTE_SAFE(cudaMemcpy(host.Grid, device.Grid, host.GetSize(), cudaMemcpyDeviceToHost));

		ERR(host.Grid[0]);
	}
}