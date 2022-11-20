#ifndef KERNEL_H
#define KERNEL_H

#include "Core/Math/Math.h"
#include "Core/Math/Scalar3f8.h"

namespace vfd {
	class CubicKernel {
	public:
		static float GetRadius() { 
			return m_Radius;
		}

		static void SetRadius(const float value) {
			m_Radius = value;
			constexpr float pi = static_cast<float>(PI);

			const float h3 = m_Radius * m_Radius * m_Radius;
			m_K = static_cast<float>(8.0) / (pi * h3);
			m_L = static_cast<float>(48.0) / (pi * h3);
			m_WZero = W({ 0.0f, 0.0f, 0.0f });
		}

		static float W(const float r) {
			float res = 0.0;
			const float q = r / m_Radius;
			if (q <= 1.0f) {
				if (q <= 0.5f) {
					const float q2 = q * q;
					const float q3 = q2 * q;
					res = m_K * (static_cast<float>(6.0) * q3 - static_cast<float>(6.0) * q2 + static_cast<float>(1.0));
				}
				else {
					res = m_K * (static_cast<float>(2.0) * pow(static_cast<float>(1.0) - q, static_cast<float>(3.0)));
				}
			}
			return res;
		}

		static float W(const glm::vec3& r) {
			return W(sqrt(dot(r, r)));
		}

		static glm::vec3 GradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = sqrt(dot(r, r));
			const float q = rl / m_Radius;
			if (rl > 1.0e-5 && (q <= 1.0f))
			{
				const glm::vec3 gradientQ = r * (static_cast<float>(1.0) / (rl * m_Radius));
				if (q <= 0.5f)
				{
					res = m_L * q * ((float)3.0 * q - static_cast<float>(2.0)) * gradientQ;
				}
				else
				{
					const float factor = static_cast<float>(1.0) - q;
					res = m_L * (-factor * factor) * gradientQ;
				}
			}
			else {
				res = { 0.0f, 0.0f, 0.0f };
			}
			return res;
		}

		static float WZero() {
			return m_WZero;
		}
	protected:
		inline static float m_Radius;
		inline static float m_K;
		inline static float m_L;
		inline static float m_WZero;
	};

	template <typename KernelType, unsigned int Resolution = 10000u>
	class PrecomputedKernel {
	public:
		static float GetRadius() { 
			return m_Radius; 
		}

		static void SetRadius(const float value) {
			m_Radius = value;
			m_Radius2 = m_Radius * m_Radius;
			KernelType::SetRadius(value);
			const float stepSize = m_Radius / (float)(Resolution - 1);
			m_InvStepSize = static_cast<float>(1.0) / stepSize;
			for (unsigned int i = 0; i < Resolution; i++)
			{
				const float posX = stepSize * (float)i;
				m_W[i] = KernelType::W(posX);
				KernelType::SetRadius(value);
				if (posX > 1.0e-9) {
					m_GradientW[i] = KernelType::GradientW(glm::vec3(posX, 0.0f, 0.0f ))[0] / posX;
				}
				else {
					m_GradientW[i] = 0.0f;
				}
			}
			m_GradientW[Resolution] = 0.0f;
			m_WZero = W(static_cast<float>(0.0));
		}

		static float W(const float r) {
			float res = 0.0f;
			if (r <= m_Radius) {
				const unsigned int pos = std::min<unsigned int>((unsigned int)(r * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}
			return res;
		}

		static float W(const glm::vec3& r) {
			float res = 0.0;
			const float r2 = dot(r, r);
			if (r2 < m_Radius2) {
				const float rl = sqrt(r2);
				const unsigned int pos = std::min<unsigned int>((unsigned int)(rl * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}
			return res;
		}

		static glm::vec3 GradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = sqrt(glm::dot(r, r));
			if (rl <= m_Radius) {
				const unsigned int pos = std::min<unsigned int>(static_cast<unsigned int>(rl * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_GradientW[pos] + m_GradientW[pos + 1]) * r;
			}
			else {
				res = { 0.0f, 0.0f, 0.0f };
			}

			return res;
		}

		static float WZero() {
			return m_WZero;
		}

	protected:
		inline static float m_W[Resolution];
		inline static float m_GradientW[Resolution + 1];
		inline static float m_Radius;
		inline static float m_Radius2;
		inline static float m_InvStepSize;
		inline static float m_WZero;
	};

	class CubicKernelAVX {
	public:
		static float GetRadius() { return m_r; }

		static void SetRadius(const float value)
		{
			m_r = value;
			m_invRadius = Scalar8(1.0f / value);
			m_invRadius2 = m_invRadius * m_invRadius;
			const float pi = static_cast<float>(PI);

			const float h3 = m_r * m_r * m_r;
			m_k = Scalar8(8.0f / (pi * h3));
			m_l = Scalar8(48.0f / (pi * h3));
			
			m_zero = Scalar8(0.0f);
			m_half = Scalar8(0.5f);
			m_one = Scalar8(1.0f);
			m_eps = Scalar8(1.0e-5f);
			Scalar8 WZero = W(m_zero);
			float tmp[8];
			WZero.Store(tmp);
			m_WZero = tmp[0];
		}

		static Scalar8 W(const Scalar8 r)
		{
			const Scalar8 q = r * m_invRadius;
			const Scalar8 v = m_one - q;
			const Scalar8 res1 = m_k * (Scalar8(-6.0f) * q * q * v + m_one);
			const Scalar8 res2 = (m_k * Scalar8(2.0f) * (v * v * v));

			Scalar8 res = Blend(q <= m_one, res2, m_zero);
			res = Blend(q <= m_half, res1, res);

			return res;
		}

		static Scalar8 W(const Scalar3f8& r)
		{
			return W(r.Norm());
		}

		static Scalar3f8 GradientW(const Scalar3f8& r)
		{
			const Scalar8 rl = r.Norm();
			const Scalar8 q = rl * m_invRadius;

			const Scalar8 res1 = (m_l * m_invRadius2 * (MultiplyAndSubtract(Scalar8(3.0f), q, Scalar8(2.0f))));

			const Scalar8 v = m_one - q;
			const Scalar8 gradq = (m_invRadius / rl);
			const Scalar8 res2 = gradq * (-m_l * (v * v));

			Scalar8 res = Blend(q <= m_one, res2, m_zero);
			res = Blend(q <= m_half, res1, res);
			res = Blend(rl > m_eps, res, m_zero);

			return r * res;
		}

		static const float& WZero()
		{
			return m_WZero;
		}

	protected:
		inline static float m_r;
		inline static Scalar8 m_invRadius;
		inline static Scalar8 m_invRadius2;
		inline static Scalar8 m_k;
		inline static Scalar8 m_l;
		inline static float m_WZero;
		inline static Scalar8 m_zero;
		inline static Scalar8 m_half;
		inline static Scalar8 m_one;
		inline static Scalar8 m_eps;
	};
}

#endif // !KERNEL_H