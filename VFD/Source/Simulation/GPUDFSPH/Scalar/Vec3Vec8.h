#ifndef VEC_3_VEC_8_H
#define VEC_3_VEC_8_H

#include "pch.h"
#include "Vec8.h"

namespace vfd {
	class vec3vec8 {
	public:
		__host__ __device__ vec3vec8() = default;
		__host__ __device__ vec3vec8(const glm::vec3& x) {
			Value[0].Set(x[0]);
			Value[1].Set(x[1]);
			Value[2].Set(x[2]);
		}

		__host__ __device__ vec8& X() {
			return Value[0];
		}

		__host__ __device__ vec8& Y() {
			return Value[1];
		}

		__host__ __device__ vec8& Z() {
			return Value[2];
		}

		__host__ __device__ const vec8& X() const {
			return Value[0];
		}

		__host__ __device__ const vec8& Y() const {
			return Value[1];
		}

		__host__ __device__ const vec8& Z() const {
			return Value[2];
		}

		__host__ __device__ vec8 Norm() const
		{
			auto a = Value[2] * Value[2];
			auto b = MultiplyAndAdd(Value[1], Value[1], a);
			auto c = MultiplyAndAdd(Value[0], Value[0], b);

			vec8 temp;

			for (size_t i = 0; i < 8; i++)
			{
				temp.Value[i] = sqrt(c.Value[i]);
			}

			return temp;
		}

		__host__ __device__ vec8 SquaredNorm() {
			auto a = Value[2] * Value[2];
			auto b = MultiplyAndAdd(Value[1], Value[1], a);
			return MultiplyAndAdd(Value[0], Value[0], b);
		}

		__host__ __device__ vec8 Dot(const vec3vec8& a) {
			auto x = Value[2] * a.Value[2];
			auto y = MultiplyAndAdd(Value[1], a.Value[1], x);
			return MultiplyAndAdd(Value[0], a.Value[0], y);
		}

		__host__ __device__ const vec8& operator[] (int i) const {
			return Value[i];
		}
	public:
		vec8 Value[3];
	};

	__host__ __device__ inline vec3vec8 operator - (vec3vec8 const& a, vec3vec8 const& b) {
		vec3vec8 res;
		res.Value[0] = a.Value[0] - b.Value[0];
		res.Value[1] = a.Value[1] - b.Value[1];
		res.Value[2] = a.Value[2] - b.Value[2];
		return res;
	}

	__host__ __device__ inline vec3vec8 operator + (vec3vec8 const& a, vec3vec8 const& b) {
		vec3vec8 res;
		res.Value[0] = a.Value[0] + b.Value[0];
		res.Value[1] = a.Value[1] + b.Value[1];
		res.Value[2] = a.Value[2] + b.Value[2];
		return res;
	}

	__host__ __device__ inline vec3vec8 operator * (vec3vec8 const& a, const vec8& s) {
		vec3vec8 res;
		res.Value[0] = a.Value[0] * s;
		res.Value[1] = a.Value[1] * s;
		res.Value[2] = a.Value[2] * s;
		return res;
	}

	__host__ __device__ inline vec3vec8 ConvertScalarZeroVec8(const unsigned int* idx, const float* v, const unsigned char count = 8u) {
		vec3vec8 x;

		switch (count)
		{
		case 1u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 2u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 3u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 4u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 5u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], 0.0f, 0.0f, 0.0f);
			break;
		case 6u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], v[3 * idx[5] + 0], 0.0f, 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], v[3 * idx[5] + 1], 0.0f, 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], v[3 * idx[5] + 2], 0.0f, 0.0f);
			break;
		case 7u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], v[3 * idx[5] + 0], v[3 * idx[6] + 0], 0.0f);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], v[3 * idx[5] + 1], v[3 * idx[6] + 1], 0.0f);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], v[3 * idx[5] + 2], v[3 * idx[6] + 2], 0.0f);
			break;
		case 8u:
			x.Value[0] = vec8(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], v[3 * idx[5] + 0], v[3 * idx[6] + 0], v[3 * idx[7] + 0]);
			x.Value[1] = vec8(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], v[3 * idx[5] + 1], v[3 * idx[6] + 1], v[3 * idx[7] + 1]);
			x.Value[2] = vec8(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], v[3 * idx[5] + 2], v[3 * idx[6] + 2], v[3 * idx[7] + 2]);
		}
		return x;
	}

	__host__ __device__ inline vec3vec8 ConvertScalarZeroVec8(const unsigned int* idx, const glm::vec3* v, const unsigned char count = 8u) {
		vec3vec8 x;
		switch (count)
		{
		case 1u:
			x.Value[0] = vec8(v[idx[0]][0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 2u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 3u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 4u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], 0.0f, 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 5u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], 0.0f, 0.0f, 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], 0.0f, 0.0f, 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], 0.0f, 0.0f, 0.0f);
			break;
		case 6u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], v[idx[5]][0], 0.0f, 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], v[idx[5]][1], 0.0f, 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], v[idx[5]][2], 0.0f, 0.0f);
			break;
		case 7u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], v[idx[5]][0], v[idx[6]][0], 0.0f);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], v[idx[5]][1], v[idx[6]][1], 0.0f);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], v[idx[5]][2], v[idx[6]][2], 0.0f);
			break;
		case 8u:
			x.Value[0] = vec8(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], v[idx[5]][0], v[idx[6]][0], v[idx[7]][0]);
			x.Value[1] = vec8(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], v[idx[5]][1], v[idx[6]][1], v[idx[7]][1]);
			x.Value[2] = vec8(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], v[idx[5]][2], v[idx[6]][2], v[idx[7]][2]);
		}

		return x;
	}
}

#endif // !VEC_3_VEC_8_H