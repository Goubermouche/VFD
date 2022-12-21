#ifndef VEC_8_H
#define VEC_8_H

namespace vfd
{
	class vec8
	{
	public:
		__host__ __device__ vec8() = default;
		__host__ __device__ vec8(float v)
		{
			for (int i = 0; i < 8; ++i)
			{
				Value[i] = v;
			}
		}

		__host__ __device__ vec8(const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6, const float f7) {
			Value[0] = f0;
			Value[1] = f1;
			Value[2] = f2;
			Value[3] = f3;
			Value[4] = f4;
			Value[5] = f5;
			Value[6] = f6;
			Value[7] = f7;
		}

		__host__ __device__ vec8& operator = (const float(&v)[8]) {
			for (int i = 0; i < 8; ++i)
			{
				Value[i] = v[i];
			}

			return *this;
		}

		__host__ __device__ float Reduce()
		{
			return Value[0] + Value[1] + Value[2] + Value[3] + Value[4] + Value[5] + Value[6] + Value[7];
		}

		__host__ __device__ void Set(float v) {
			for (int i = 0; i < 8; ++i)
			{
				Value[i] = v;
			}
		}
	public:
		float Value[8];
	};

	__host__ __device__ inline vec8 operator - (vec8& a) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = -a.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8 operator - (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = a.Value[i] - b.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8& operator -= (vec8& a, vec8 const& b) {
		for (size_t i = 0; i < 8; i++)
		{
			a.Value[i] -= b.Value[i];
		}

		return a;
	}

	__host__ __device__ static vec8 operator + (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = a.Value[i] + b.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8& operator += (vec8& a, vec8 const& b) {
		for (size_t i = 0; i < 8; i++)
		{
			a.Value[i] += b.Value[i];
		}

		return a;
	}

	__host__ __device__ static vec8 operator * (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = a.Value[i] * b.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8 operator / (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = a.Value[i] / b.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8 operator > (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = static_cast<float>(a.Value[i] > b.Value[i]);
		}

		return temp;
	}

	__host__ __device__ static vec8 operator <= (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = static_cast<float>(a.Value[i] <= b.Value[i]);
		}

		return temp;
	}

	__host__ __device__ static vec8 Blend(vec8 const& c, vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = c.Value[i] ? a.Value[i] : b.Value[i];
		}
	}

	__host__ __device__ static vec8 MultiplyAndSubtract(const vec8& a, const vec8& b, const vec8& c)
	{
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = a.Value[i] * b.Value[i] - c.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8 MultiplyAndAdd(const vec8& a, const vec8& b, const vec8& c)
	{
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.Value[i] = a.Value[i] * b.Value[i] + c.Value[i];
		}

		return temp;
	}

	__host__ __device__ static vec8 ConvertZeroVec8(const unsigned int* idx, const float* x, const unsigned char count = 8u)
	{
		switch (count)
		{
		case 1u:
			return vec8(x[idx[0]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 2u:
			return vec8(x[idx[0]], x[idx[1]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 3u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 4u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], 0.0f, 0.0f, 0.0f, 0.0f);
		case 5u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], 0.0f, 0.0f, 0.0f);
		case 6u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], 0.0f, 0.0f);
		case 7u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], x[idx[6]], 0.0f);
		case 8u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], x[idx[6]], x[idx[7]]);
		}
	}

	__host__ __device__ static vec8 ConvertZeroVec8(const float x, const unsigned char count = 8u)
	{
		switch (count)
		{
		case 1u:
			return vec8(x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 2u:
			return vec8(x, x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 3u:
			return vec8(x, x, x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 4u:
			return vec8(x, x, x, x, 0.0f, 0.0f, 0.0f, 0.0f);
		case 5u:
			return vec8(x, x, x, x, x, 0.0f, 0.0f, 0.0f);
		case 6u:
			return vec8(x, x, x, x, x, x, 0.0f, 0.0f);
		case 7u:
			return vec8(x, x, x, x, x, x, x, 0.0f);
		case 8u:
			return vec8(x, x, x, x, x, x, x, x);
		}
	}
}

#endif VEC_8_H