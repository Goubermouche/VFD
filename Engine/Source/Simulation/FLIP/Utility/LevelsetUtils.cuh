#ifndef LEVELSET_UTILS_CUH
#define LEVELSET_UTILS_CUH

namespace fe {
    template <class T>
    static void _swap(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    template<class T>
    static __host__ void _sort(T& a, T& b, T& c, T& d)
    {
        if (a > b) { _swap(a, b); }
        if (c > d) { _swap(c, d); }
        if (a > c) { _swap(a, c); }
        if (b > d) { _swap(b, d); }
        if (b > c) { _swap(b, c); }
    }

    // Assumes phi0<0 and phi1>=0, phi2>=0, and phi3>=0 or vice versa.
// In particular, phi0 must not equal any of phi1, phi2 or phi3.
    template<class T>
    static __host__ T _sortedTetFraction(T phi0, T phi1, T phi2, T phi3) {
        return phi0 * phi0 * phi0 / ((phi0 - phi1) * (phi0 - phi2) * (phi0 - phi3));
    }

    // Assumes phi0<0, phi1<0, and phi2>=0, and phi3>=0 or vice versa.
    // In particular, phi0 and phi1 must not equal any of phi2 and phi3.
    template<class T>
    static __host__ T _sortedPrismFraction(T phi0, T phi1, T phi2, T phi3) {
        T a = phi0 / (phi0 - phi2);
        T b = phi0 / (phi0 - phi3);
        T c = phi1 / (phi1 - phi3);
        T d = phi1 / (phi1 - phi2);
        return a * b * (1 - d) + b * (1 - c) * d + c * d;
    }

    static __host__ float volumeFraction(float phi0, float phi1, float phi2, float phi3) {
        _sort(phi0, phi1, phi2, phi3);
        if (phi3 <= 0) {
            return 1;
        }
        else if (phi2 <= 0) {
            return 1 - _sortedTetFraction(phi3, phi2, phi1, phi0);
        }
        else if (phi1 <= 0) {
            return _sortedPrismFraction(phi0, phi1, phi2, phi3);
        }
        else if (phi0 <= 0) {
            return _sortedTetFraction(phi0, phi1, phi2, phi3);
        }
        else {
            return 0;
        }
    }

    static __host__ float VolumeFraction(float phi000, float phi100,
        float phi010, float phi110,
        float phi001, float phi101,
        float phi011, float phi111) {
        // This is the average of the two possible decompositions of the cube into
        // five tetrahedra.
        return (volumeFraction(phi000, phi001, phi101, phi011) +
            volumeFraction(phi000, phi101, phi100, phi110) +
            volumeFraction(phi000, phi010, phi011, phi110) +
            volumeFraction(phi101, phi011, phi111, phi110) +
            2 * volumeFraction(phi000, phi011, phi101, phi110) +
            volumeFraction(phi100, phi101, phi001, phi111) +
            volumeFraction(phi100, phi001, phi000, phi010) +
            volumeFraction(phi100, phi110, phi111, phi010) +
            volumeFraction(phi001, phi111, phi011, phi010) +
            2 * volumeFraction(phi100, phi111, phi001, phi010)) / 12.0f;
    }
}

#endif // !LEVELSET_UTILS_CUH
