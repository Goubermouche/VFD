#include "pch.h"
#include "PointSetP.h"

#include "PointSetImplementationP.h"

namespace vfd {
	PointSet::PointSet(const PointSet& other) {
		this->m_Dynamic = other.m_Dynamic;
		this->m_Points = other.m_Points;
		this->m_PointCount = other.m_PointCount;
		this->m_UserData = other.m_UserData;
		this->m_SortedIndices = other.m_SortedIndices;
		this->m_Neighbors = other.m_Neighbors;
		
		PointSetImplementation* ptr = other.m_Implementation.get();
		m_Implementation = std::make_unique<PointSetImplementation>(PointSetImplementation(*ptr));
	}

	PointSet::PointSet(DFSPHParticle const* x, std::size_t n, bool dynamic, void* userData)
		: m_Points(x), m_PointCount(n), m_Dynamic(dynamic), m_UserData(userData)
	{
		m_Implementation = std::make_unique<PointSetImplementation>(n, (DFSPHParticle*)x);
	}

	void PointSet::Resize(const DFSPHParticle* x, std::size_t n)
	{
		m_Points = x;
		m_PointCount = n;
		m_Implementation->Resize(n, (DFSPHParticle*)x);
	}
}