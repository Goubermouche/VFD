#ifndef VEC_8_H
#define VEC_8_H

namespace vfd
{
	class vec8
	{
	public:
		vec8() = default;
		vec8(float v)
		{
			for (int i = 0; i < 8; ++i)
			{
				value[i] = v;
			}
		}

		vec8(const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6, const float f7) {
			value[0] = f0;
			value[1] = f1;
			value[2] = f2;
			value[3] = f3;
			value[4] = f4;
			value[5] = f5;
			value[6] = f6;
			value[7] = f7;
		}

		vec8& operator = (const float(&v)[8]) {
			for (int i = 0; i < 8; ++i)
			{
				value[i] = v[i];
			}

			return *this;
		}

		float Reduce()
		{
			return value[0] + value[1] + value[2] + value[3] + value[4] + value[5] + value[6] + value[7];
		}
	public:
		float value[8];
	};

	inline vec8 operator - (vec8& a) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = -a.value[i];
		}

		return temp;
	}

	static vec8 operator - (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = a.value[i] - b.value[i];
		}

		return temp;
	}

	static vec8& operator -= (vec8& a, vec8 const& b) {
		for (size_t i = 0; i < 8; i++)
		{
			a.value[i] -= b.value[i];
		}

		return a;
	}

	static vec8 operator + (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = a.value[i] + b.value[i];
		}

		return temp;
	}

	static vec8& operator += (vec8& a, vec8 const& b) {
		for (size_t i = 0; i < 8; i++)
		{
			a.value[i] += b.value[i];
		}

		return a;
	}

	static vec8 operator * (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = a.value[i] * b.value[i];
		}

		return temp;
	}

	static vec8 operator / (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = a.value[i] / b.value[i];
		}

		return temp;
	}

	static vec8 operator > (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = static_cast<float>(a.value[i] > b.value[i]);
		}

		return temp;
	}

	static vec8 operator <= (vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = static_cast<float>(a.value[i] <= b.value[i]);
		}

		return temp;
	}

	static vec8 Blend(vec8 const& c, vec8 const& a, vec8 const& b) {
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = c.value[i] ? a.value[i] : b.value[i];
		}
	}

	static vec8 MultiplyAndSubtract(const vec8& a, const vec8& b, const vec8& c)
	{
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = a.value[i] * b.value[i] - c.value[i];
		}

		return temp;
	}

	static vec8 MultiplyAndAdd(const vec8& a, const vec8& b, const vec8& c)
	{
		vec8 temp;

		for (size_t i = 0; i < 8; i++)
		{
			temp.value[i] = a.value[i] * b.value[i] + c.value[i];
		}

		return temp;
	}

	static vec8 ConvertZeroVec8(const unsigned int* idx, const float* x, const unsigned char count = 8u)
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

	static vec8 ConvertZeroVec8(const float x, const unsigned char count = 8u)
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

	static vec8 ConvertZeroVec8(const unsigned int* idx, const float* x, const unsigned char count = 8u)
	{
		switch (count)
		{
		case 1u:
			return vec8(x[idx[0]], 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
		case 2u:
			return vec8(x[idx[0]], x[idx[1]], 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
		case 3u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
		case 4u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], 1.0f, 1.0f, 1.0f, 1.0f);
		case 5u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], 1.0f, 1.0f, 1.0f);
		case 6u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], 1.0f, 1.0f);
		case 7u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], x[idx[6]], 1.0f);
		case 8u:
			return vec8(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], x[idx[6]], x[idx[7]]);
		}
	}
}

#endif VEC_8_H