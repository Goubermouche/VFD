#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void ClearAccelerationsKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Acceleration = info->Gravity;
}

__global__ void CalculateVelocitiesKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Velocity += info->TimeStepSize * particles[i].Acceleration;
}

__global__ void CalculatePositionsKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Position += info->TimeStepSize * particles[i].Velocity;
}

__device__ unsigned int MultiToSingleIndex(vfd::RigidBodyData* rigidBody, const glm::uvec3& index)
{
	return rigidBody->Resolution.y * rigidBody->Resolution.x * index.z + rigidBody->Resolution.x * index.y + index.x;
}

__device__ glm::uvec3 SingleToMultiIndex(vfd::RigidBodyData* rigidBody, const unsigned int index)
{
	const unsigned int n01 = rigidBody->Resolution.x * rigidBody->Resolution.y;
	unsigned int k = index / n01;
	const unsigned int temp = index % n01;
	double j = temp / rigidBody->Resolution.x;
	double i = temp % rigidBody->Resolution.x;

	return glm::uvec3(i, j, k);
}

__device__ vfd::BoundingBox<glm::dvec3> CalculateSubDomain(vfd::RigidBodyData* rigidBody, const glm::uvec3& index)
{
	const glm::dvec3 origin = rigidBody->Domain.min + ((glm::dvec3)index * rigidBody->CellSize);
	vfd::BoundingBox<glm::dvec3> box;
	box.min = origin;
	box.max = origin + rigidBody->CellSize;
	return box;
}

__device__ vfd::BoundingBox<glm::dvec3> CalculateSubDomain(vfd::RigidBodyData* rigidBody, unsigned int index)
{
	return CalculateSubDomain(rigidBody, SingleToMultiIndex(rigidBody, index));
}

__device__ void ShapeFunction(double(&res)[32], const glm::dvec3& xi, glm::dvec3(&gradient)[32])
{
	const double x = xi[0];
	const double y = xi[1];
	const double z = xi[2];

	const double x2 = x * x;
	const double y2 = y * y;
	const double z2 = z * z;

	const double _1mx = 1.0 - x;
	const double _1my = 1.0 - y;
	const double _1mz = 1.0 - z;

	const double _1px = 1.0 + x;
	const double _1py = 1.0 + y;
	const double _1pz = 1.0 + z;

	const double _1m3x = 1.0 - 3.0 * x;
	const double _1m3y = 1.0 - 3.0 * y;
	const double _1m3z = 1.0 - 3.0 * z;

	const double _1p3x = 1.0 + 3.0 * x;
	const double _1p3y = 1.0 + 3.0 * y;
	const double _1p3z = 1.0 + 3.0 * z;

	const double _1mxt1my = _1mx * _1my;
	const double _1mxt1py = _1mx * _1py;
	const double _1pxt1my = _1px * _1my;
	const double _1pxt1py = _1px * _1py;

	const double _1mxt1mz = _1mx * _1mz;
	const double _1mxt1pz = _1mx * _1pz;
	const double _1pxt1mz = _1px * _1mz;
	const double _1pxt1pz = _1px * _1pz;

	const double _1myt1mz = _1my * _1mz;
	const double _1myt1pz = _1my * _1pz;
	const double _1pyt1mz = _1py * _1mz;
	const double _1pyt1pz = _1py * _1pz;

	const double _1mx2 = 1.0 - x2;
	const double _1my2 = 1.0 - y2;
	const double _1mz2 = 1.0 - z2;

	// Corner nodes.
	double fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
	res[0] = fac * _1mxt1my * _1mz;
	res[1] = fac * _1pxt1my * _1mz;
	res[2] = fac * _1mxt1py * _1mz;
	res[3] = fac * _1pxt1py * _1mz;
	res[4] = fac * _1mxt1my * _1pz;
	res[5] = fac * _1pxt1my * _1pz;
	res[6] = fac * _1mxt1py * _1pz;
	res[7] = fac * _1pxt1py * _1pz;

	// Edge nodes.
	fac = 9.0 / 64.0 * _1mx2;
	const double fact1m3x = fac * _1m3x;
	const double fact1p3x = fac * _1p3x;
	res[8] = fact1m3x * _1myt1mz;
	res[9] = fact1p3x * _1myt1mz;
	res[10] = fact1m3x * _1myt1pz;
	res[11] = fact1p3x * _1myt1pz;
	res[12] = fact1m3x * _1pyt1mz;
	res[13] = fact1p3x * _1pyt1mz;
	res[14] = fact1m3x * _1pyt1pz;
	res[15] = fact1p3x * _1pyt1pz;

	fac = 9.0 / 64.0 * _1my2;
	const double fact1m3y = fac * _1m3y;
	const double fact1p3y = fac * _1p3y;
	res[16] = fact1m3y * _1mxt1mz;
	res[17] = fact1p3y * _1mxt1mz;
	res[18] = fact1m3y * _1pxt1mz;
	res[19] = fact1p3y * _1pxt1mz;
	res[20] = fact1m3y * _1mxt1pz;
	res[21] = fact1p3y * _1mxt1pz;
	res[22] = fact1m3y * _1pxt1pz;
	res[23] = fact1p3y * _1pxt1pz;

	fac = 9.0 / 64.0 * _1mz2;
	const double fact1m3z = fac * _1m3z;
	const double fact1p3z = fac * _1p3z;
	res[24] = fact1m3z * _1mxt1my;
	res[25] = fact1p3z * _1mxt1my;
	res[26] = fact1m3z * _1mxt1py;
	res[27] = fact1p3z * _1mxt1py;
	res[28] = fact1m3z * _1pxt1my;
	res[29] = fact1p3z * _1pxt1my;
	res[30] = fact1m3z * _1pxt1py;
	res[31] = fact1p3z * _1pxt1py;

	const double _9t3x2py2pz2m19 = 9.0 * (3.0 * x2  + y2  + z2) - 19.0;
	const double _9tx2p3y2pz2m19 = 9.0 * (x2  + 3.0 * y2  + z2) - 19.0;
	const double _9tx2py2p3z2m19 = 9.0 * (x2  + y2  + 3.0 * z2) - 19.0;
	const double _18x = 18.0 * x;
	const double _18y = 18.0 * y;
	const double _18z = 18.0 * z;

	const double _3m9x2 = 3.0 - 9.0 * x2;
	const double _3m9y2 = 3.0 - 9.0 * y2;
	const double _3m9z2 = 3.0 - 9.0 * z2;

	const double _2x = 2.0 * x;
	const double _2y = 2.0 * y;
	const double _2z = 2.0 * z;

	const double _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
	const double _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
	const double _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
	const double _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
	const double _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
	const double _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

	gradient[0][0] = _18xm9t3x2py2pz2m19 * _1myt1mz;
	gradient[0][1] = _1mxt1mz * _18ym9tx2p3y2pz2m19;
	gradient[0][2] = _1mxt1my * _18zm9tx2py2p3z2m19;
	gradient[1][0] = _18xp9t3x2py2pz2m19 * _1myt1mz;
	gradient[1][1] = _1pxt1mz * _18ym9tx2p3y2pz2m19;
	gradient[1][2] = _1pxt1my * _18zm9tx2py2p3z2m19;
	gradient[2][0] = _18xm9t3x2py2pz2m19 * _1pyt1mz;
	gradient[2][1] = _1mxt1mz * _18yp9tx2p3y2pz2m19;
	gradient[2][2] = _1mxt1py * _18zm9tx2py2p3z2m19;
	gradient[3][0] = _18xp9t3x2py2pz2m19 * _1pyt1mz;
	gradient[3][1] = _1pxt1mz * _18yp9tx2p3y2pz2m19;
	gradient[3][2] = _1pxt1py * _18zm9tx2py2p3z2m19;
	gradient[4][0] = _18xm9t3x2py2pz2m19 * _1myt1pz;
	gradient[4][1] = _1mxt1pz * _18ym9tx2p3y2pz2m19;
	gradient[4][2] = _1mxt1my * _18zp9tx2py2p3z2m19;
	gradient[5][0] = _18xp9t3x2py2pz2m19 * _1myt1pz;
	gradient[5][1] = _1pxt1pz * _18ym9tx2p3y2pz2m19;
	gradient[5][2] = _1pxt1my * _18zp9tx2py2p3z2m19;
	gradient[6][0] = _18xm9t3x2py2pz2m19 * _1pyt1pz;
	gradient[6][1] = _1mxt1pz * _18yp9tx2p3y2pz2m19;
	gradient[6][2] = _1mxt1py * _18zp9tx2py2p3z2m19;
	gradient[7][0] = _18xp9t3x2py2pz2m19 * _1pyt1pz;
	gradient[7][1] = _1pxt1pz * _18yp9tx2p3y2pz2m19;
	gradient[7][2] = _1pxt1py * _18zp9tx2py2p3z2m19;

	gradient[0][0] /= 64.0;
	gradient[0][1] /= 64.0;
	gradient[0][2] /= 64.0;
	gradient[1][0] /= 64.0;
	gradient[1][1] /= 64.0;
	gradient[1][2] /= 64.0;
	gradient[2][0] /= 64.0;
	gradient[2][1] /= 64.0;
	gradient[2][2] /= 64.0;
	gradient[3][0] /= 64.0;
	gradient[3][1] /= 64.0;
	gradient[3][2] /= 64.0;
	gradient[4][0] /= 64.0;
	gradient[4][1] /= 64.0;
	gradient[4][2] /= 64.0;
	gradient[5][0] /= 64.0;
	gradient[5][1] /= 64.0;
	gradient[5][2] /= 64.0;
	gradient[6][0] /= 64.0;
	gradient[6][1] /= 64.0;
	gradient[6][2] /= 64.0;
	gradient[7][0] /= 64.0;
	gradient[7][1] /= 64.0;
	gradient[7][2] /= 64.0;

	const double _m3m9x2m2x = -_3m9x2 - _2x;
	const double _p3m9x2m2x =  _3m9x2 - _2x;
	const double _1mx2t1m3x =  _1mx2  * _1m3x;
	const double _1mx2t1p3x =  _1mx2  * _1p3x;
	gradient[8][0 ] =  _m3m9x2m2x * _1myt1mz, gradient[8][1 ] = -_1mx2t1m3x * _1mz,    gradient[8][2 ] = -_1mx2t1m3x * _1my;
	gradient[9][0 ] =  _p3m9x2m2x * _1myt1mz, gradient[9][1 ] = -_1mx2t1p3x * _1mz,    gradient[9][2 ] = -_1mx2t1p3x * _1my;
	gradient[10][0] =  _m3m9x2m2x * _1myt1pz, gradient[10][1] = -_1mx2t1m3x * _1pz,    gradient[10][2] =  _1mx2t1m3x * _1my;
	gradient[11][0] =  _p3m9x2m2x * _1myt1pz, gradient[11][1] = -_1mx2t1p3x * _1pz,    gradient[11][2] =  _1mx2t1p3x * _1my;
	gradient[12][0] =  _m3m9x2m2x * _1pyt1mz, gradient[12][1] =  _1mx2t1m3x * _1mz,    gradient[12][2] = -_1mx2t1m3x * _1py;
	gradient[13][0] =  _p3m9x2m2x * _1pyt1mz, gradient[13][1] =  _1mx2t1p3x * _1mz,    gradient[13][2] = -_1mx2t1p3x * _1py;
	gradient[14][0] =  _m3m9x2m2x * _1pyt1pz, gradient[14][1] =  _1mx2t1m3x * _1pz,    gradient[14][2] =  _1mx2t1m3x * _1py;
	gradient[15][0] =  _p3m9x2m2x * _1pyt1pz, gradient[15][1] =  _1mx2t1p3x * _1pz,    gradient[15][2] =  _1mx2t1p3x * _1py;

	const double _m3m9y2m2y = -_3m9y2 - _2y;
	const double _p3m9y2m2y =  _3m9y2 - _2y;
	const double _1my2t1m3y =  _1my2  * _1m3y;
	const double _1my2t1p3y =  _1my2  * _1p3y;
	gradient[16][0] = -_1my2t1m3y * _1mz,     gradient[16][1] = _m3m9y2m2y * _1mxt1mz, gradient[16][2] = -_1my2t1m3y * _1mx;
	gradient[17][0] = -_1my2t1p3y * _1mz,     gradient[17][1] = _p3m9y2m2y * _1mxt1mz, gradient[17][2] = -_1my2t1p3y * _1mx;
	gradient[18][0] =  _1my2t1m3y * _1mz,     gradient[18][1] = _m3m9y2m2y * _1pxt1mz, gradient[18][2] = -_1my2t1m3y * _1px;
	gradient[19][0] =  _1my2t1p3y * _1mz,     gradient[19][1] = _p3m9y2m2y * _1pxt1mz, gradient[19][2] = -_1my2t1p3y * _1px;
	gradient[20][0] = -_1my2t1m3y * _1pz,     gradient[20][1] = _m3m9y2m2y * _1mxt1pz, gradient[20][2] =  _1my2t1m3y * _1mx;
	gradient[21][0] = -_1my2t1p3y * _1pz,     gradient[21][1] = _p3m9y2m2y * _1mxt1pz, gradient[21][2] =  _1my2t1p3y * _1mx;
	gradient[22][0] =  _1my2t1m3y * _1pz,     gradient[22][1] = _m3m9y2m2y * _1pxt1pz, gradient[22][2] =  _1my2t1m3y * _1px;
	gradient[23][0] =  _1my2t1p3y * _1pz,     gradient[23][1] = _p3m9y2m2y * _1pxt1pz, gradient[23][2] =  _1my2t1p3y * _1px;

	const double _m3m9z2m2z = -_3m9z2 - _2z;
	const double _p3m9z2m2z =  _3m9z2 - _2z;
	const double _1mz2t1m3z =  _1mz2  * _1m3z;
	const double _1mz2t1p3z =  _1mz2  * _1p3z;
	gradient[24][0] = -_1mz2t1m3z * _1my,     gradient[24][1] = -_1mz2t1m3z * _1mx,    gradient[24][2] = _m3m9z2m2z * _1mxt1my;
	gradient[25][0] = -_1mz2t1p3z * _1my,     gradient[25][1] = -_1mz2t1p3z * _1mx,    gradient[25][2] = _p3m9z2m2z * _1mxt1my;
	gradient[26][0] = -_1mz2t1m3z * _1py,     gradient[26][1] =  _1mz2t1m3z * _1mx,    gradient[26][2] = _m3m9z2m2z * _1mxt1py;
	gradient[27][0] = -_1mz2t1p3z * _1py,     gradient[27][1] =  _1mz2t1p3z * _1mx,    gradient[27][2] = _p3m9z2m2z * _1mxt1py;
	gradient[28][0] =  _1mz2t1m3z * _1my,     gradient[28][1] = -_1mz2t1m3z * _1px,    gradient[28][2] = _m3m9z2m2z * _1pxt1my;
	gradient[29][0] =  _1mz2t1p3z * _1my,     gradient[29][1] = -_1mz2t1p3z * _1px,    gradient[29][2] = _p3m9z2m2z * _1pxt1my;
	gradient[30][0] =  _1mz2t1m3z * _1py,     gradient[30][1] =  _1mz2t1m3z * _1px,    gradient[30][2] = _m3m9z2m2z * _1pxt1py;
	gradient[31][0] =  _1mz2t1p3z * _1py,     gradient[31][1] =  _1mz2t1p3z * _1px,    gradient[31][2] = _p3m9z2m2z * _1pxt1py;

	const double rfe = 9.0 / 64.0;
	gradient[31][0] *= rfe;
	gradient[31][1] *= rfe;
	gradient[31][2] *= rfe;
	gradient[30][0] *= rfe;
	gradient[30][1] *= rfe;
	gradient[30][2] *= rfe;
	gradient[29][0] *= rfe;
	gradient[29][1] *= rfe;
	gradient[29][2] *= rfe;
	gradient[28][0] *= rfe;
	gradient[28][1] *= rfe;
	gradient[28][2] *= rfe;
	gradient[27][0] *= rfe;
	gradient[27][1] *= rfe;
	gradient[27][2] *= rfe;
	gradient[26][0] *= rfe;
	gradient[26][1] *= rfe;
	gradient[26][2] *= rfe;
	gradient[25][0] *= rfe;
	gradient[25][1] *= rfe;
	gradient[25][2] *= rfe;
	gradient[24][0] *= rfe;
	gradient[24][1] *= rfe;
	gradient[24][2] *= rfe;
	gradient[23][0] *= rfe;
	gradient[23][1] *= rfe;
	gradient[23][2] *= rfe;
	gradient[22][0] *= rfe;
	gradient[22][1] *= rfe;
	gradient[22][2] *= rfe;
	gradient[21][0] *= rfe;
	gradient[21][1] *= rfe;
	gradient[21][2] *= rfe;
	gradient[20][0] *= rfe;
	gradient[20][1] *= rfe;
	gradient[20][2] *= rfe;
	gradient[19][0] *= rfe;
	gradient[19][1] *= rfe;
	gradient[19][2] *= rfe;
	gradient[18][0] *= rfe;
	gradient[18][1] *= rfe;
	gradient[18][2] *= rfe;
	gradient[17][0] *= rfe;
	gradient[17][1] *= rfe;
	gradient[17][2] *= rfe;
	gradient[16][0] *= rfe;
	gradient[16][1] *= rfe;
	gradient[16][2] *= rfe;
	gradient[15][0] *= rfe;
	gradient[15][1] *= rfe;
	gradient[15][2] *= rfe;
	gradient[14][0] *= rfe;
	gradient[14][1] *= rfe;
	gradient[14][2] *= rfe;
	gradient[13][0] *= rfe;
	gradient[13][1] *= rfe;
	gradient[13][2] *= rfe;
	gradient[12][0] *= rfe;
	gradient[12][1] *= rfe;
	gradient[12][2] *= rfe;
	gradient[11][0] *= rfe;
	gradient[11][1] *= rfe;
	gradient[11][2] *= rfe;
	gradient[10][0] *= rfe;
	gradient[10][1] *= rfe;
	gradient[10][2] *= rfe;
	gradient[9][0 ] *= rfe;
	gradient[9][1 ] *= rfe;
	gradient[9][2 ] *= rfe;
	gradient[8][0 ] *= rfe;
	gradient[8][1 ] *= rfe;
	gradient[8][2 ] *= rfe;
}

__device__ bool DetermineShapeFunctions(vfd::RigidBodyData* rigidBody, unsigned int fieldID, const glm::dvec3& x, unsigned int(&cell)[32], glm::dvec3& c0, double(&N)[32], glm::dvec3(&dN)[32])
{
	if (!rigidBody->Domain.Contains(x)) {
		return false;
	}

	glm::uvec3 mi = (rigidBody->CellSizeInverse * (x - rigidBody->Domain.min));

	if (mi[0] >= rigidBody->Resolution[0]) {
		mi[0] = rigidBody->Resolution[0] - 1;
	}

	if (mi[1] >= rigidBody->Resolution[1]) {
		mi[1] = rigidBody->Resolution[1] - 1;
	}

	if (mi[2] >= rigidBody->Resolution[2]) {
		mi[2] = rigidBody->Resolution[2] - 1;
	}

	unsigned int i = MultiToSingleIndex(rigidBody, mi);
	unsigned int i_ = rigidBody->GetCellMap(fieldID, i);
	if (i_ == UINT_MAX) {
		return false;
	}

	vfd::BoundingBox<glm::dvec3> sd = CalculateSubDomain(rigidBody, i);
	i = i_;

	const glm::dvec3 denom = sd.max - sd.min;
	c0 = 2.0 / denom;
	glm::dvec3 c1 = (sd.max + sd.min) / denom;
	glm::dvec3 xi = (c0 * x) - c1;

	#pragma unroll
	for (size_t j = 0; j < 32; j++)
	{
		cell[i] = rigidBody->GetCell(fieldID, i, j);
	}

	ShapeFunction(N, xi, dN);
	return true;
}

__device__ double Interpolate(vfd::RigidBodyData* rigidBody, unsigned int fieldID, const glm::dvec3& xi, unsigned int(&cell)[32], const glm::dvec3& c0, double(&N)[32])
{
	double phi = 0.0;

	#pragma unroll
	for (unsigned int j = 0u; j < 32u; ++j)
	{
		unsigned int v = cell[j];
		double c = rigidBody->GetNode(fieldID, v);

		if (c == DBL_MAX)
		{
			return DBL_MAX;
		}

		phi += c * N[j];
	}

	return phi;
}

__device__ double Interpolate(vfd::RigidBodyData* rigidBody, unsigned int fieldID, const glm::dvec3& xi, unsigned int(&cell)[32], const glm::dvec3& c0, double(&N)[32], glm::dvec3& gradient, glm::dvec3(&dN)[32])
{
	double phi = 0.0;
	gradient = { 0.0, 0.0, 0.0 };

	#pragma unroll
	for (unsigned int j = 0u; j < 32u; ++j)
	{
		unsigned int v = cell[j];
		double c = rigidBody->GetNode(fieldID, v);

		if (c == DBL_MAX)
		{
			gradient = { 0.0, 0.0, 0.0 };
			return DBL_MAX;
		}

		phi += c * N[j];
		gradient += c * dN[j];
	}

	gradient *= c0;

	return phi;
}

__global__ void ComputeVolumeAndBoundaryKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, vfd::RigidBodyData* rigidBody)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	//glm::dvec3 x;
	//unsigned int cell[32];
	//glm::dvec3 c0;
	//double N[32];
	//glm::dvec3 dN[32];

	//printf("%d\n", DetermineShapeFunctions(rigidBody, 0, x, cell, c0, N, dN));

	//glm::dvec3 x;
	//unsigned int cell[32];
	//glm::dvec3 c0;
	//double N[32];
	//glm::dvec3 gradient;
	//glm::dvec3 dN[32];

	//printf("%f\n", Interpolate(rigidBody, 0, x, cell, c0, N, gradient, dN));
	//printf("%f\n", Interpolate(rigidBody, 0, x, cell, c0, N));
}