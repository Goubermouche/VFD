#include "pch.h"
#include "GPUSDF.h"

#include "Compute/Utility/CUDA/cutil_math.h"
#include "GPUSDF.cuh"
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <Core/Structures/BoundingBox.h>

namespace vfd {
	GPUSDF::GPUSDF(const Ref<TriangleMesh>& mesh, int resolution)
	{
		std::vector<glm::vec3> vertList = mesh->CopyVertices();
		std::vector<glm::ivec3> faceList = mesh->CopyTriangles();

		std::vector<glm::dvec3> verts(vertList.size());
		std::vector<glm::uvec3> faces(faceList.size());

		std::transform(vertList.begin(), vertList.end(), verts.begin(), [](glm::vec3 x) { return (glm::dvec3)x; });
		std::transform(faceList.begin(), faceList.end(), faces.begin(), [](glm::ivec3 x) { return (glm::uvec3)x; });

		BoundingBox bbox(verts);

		bbox.min -= padding * dx ;
		bbox.max += padding * dx ;

        glm::uvec3 sizes = glm::uvec3(std::ceil((bbox.max[0] - bbox.min[0]) / dx),
            std::ceil((bbox.max[1] - bbox.min[1]) / dx),
            std::ceil((bbox.max[2] - bbox.min[2]) / dx));

		std::cout << "Bounding box size: (" << bbox.max.x << " " << bbox.max.y << " " << bbox.max.z << ") to (" << bbox.min.x << " " << bbox.min.y << " " << bbox.min.z << ") with dimensions " << sizes.x << " " << sizes.y << " " << sizes.z << "." << std::endl;
		std::cout << "Computing signed distance field.\n";

		MakeLevelSet3D(faces, vertList, bbox.min, dx, sizes[0], sizes[1], sizes[2], m_PHI);

        std::cout << "Processing complete.\n";

        const int width = sizes.x;
        const int height = sizes.y;
        float min = 0;
        float max = 0;

        for (size_t z = 0; z < sizes.z; z++)
        {
            uint8_t* pixels = new uint8_t[width * height * 3];

            int index = 0;
            for (int y = height - 1; y >= 0; --y)
            {
                for (int x = 0; x < width; ++x)
                {
                    float r = m_PHI(x, y, z);
                    float g = r;
                    float b = r;

                    int ir = int(255.99 * r);
                    int ig = int(255.99 * g);
                    int ib = int(255.99 * b);

                    min = std::min(min, r);
                    max = std::max(max, r);

                    pixels[index++] = ir;
                    pixels[index++] = ig;
                    pixels[index++] = ib;
                }
            }

            std::string name = "Resources/SDFTests/" + std::to_string(z) + ".jpg";
            stbi_write_jpg(name.c_str(), width, height, 3, pixels, 100);

            delete[] pixels;
        }
	}

	void GPUSDF::MakeLevelSet3D(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, const glm::vec3& origin, float dx, int ni, int nj, int nk, Array3f& phi, const int exact_band)
	{
		phi.resize(ni, nj, nk);
		phi.assign((ni + nj + nk) * dx); // upper bound on distance

		Array3i closest_tri(ni, nj, nk, -1);
		Array3i intersection_count(ni, nj, nk, 0); // intersection_count(i,j,k) is # of tri intersections in (i-1,i]x{j}x{k}
		// we begin by initializing distances near the mesh, and figuring out intersection counts
		glm::vec3 ijkmin, ijkmax;

        for (unsigned int t = 0; t < tri.size(); ++t) {
            unsigned int p, q, r; 
            p = tri[t].x;
            q = tri[t].y;
            r = tri[t].z;

            // coordinates in grid to high precision
            double fip = ((double)x[p][0] - origin[0]) / dx, fjp = ((double)x[p][1] - origin[1]) / dx, fkp = ((double)x[p][2] - origin[2]) / dx;
            double fiq = ((double)x[q][0] - origin[0]) / dx, fjq = ((double)x[q][1] - origin[1]) / dx, fkq = ((double)x[q][2] - origin[2]) / dx;
            double fir = ((double)x[r][0] - origin[0]) / dx, fjr = ((double)x[r][1] - origin[1]) / dx, fkr = ((double)x[r][2] - origin[2]) / dx;
            // do distances nearby
            int i0 = std::clamp(int(std::min(std::min(fip, fiq), fir)) - exact_band, 0, ni - 1), i1 = clamp(int(std::max(std::max(fip, fiq), fir)) + exact_band + 1, 0, ni - 1);
            int j0 = std::clamp(int(std::min(std::min(fjp, fjq), fjr)) - exact_band, 0, nj - 1), j1 = clamp(int(std::max(std::max(fjp, fjq), fjr)) + exact_band + 1, 0, nj - 1);
            int k0 = std::clamp(int(std::min(std::min(fkp, fkq), fkr)) - exact_band, 0, nk - 1), k1 = clamp(int(std::max(std::max(fkp, fkq), fkr)) + exact_band + 1, 0, nk - 1);
            for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
                glm::vec3 gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);
                float d = PointToTriangleDistance(gx, x[p], x[q], x[r]);
                if (d < phi(i, j, k)) {
                    phi(i, j, k) = d;
                    closest_tri(i, j, k) = t;
                }
            }
            // and do intersection counts
            j0 = clamp((int)std::ceil(std::min(std::min(fjp, fjq), fjr)), 0, nj - 1);
            j1 = clamp((int)std::floor(std::max(std::max(fjp, fjq), fjr)), 0, nj - 1);
            k0 = clamp((int)std::ceil(std::min(std::min(fkp, fkq), fkr)), 0, nk - 1);
            k1 = clamp((int)std::floor(std::max(std::max(fkp, fkq), fkr)), 0, nk - 1);
            for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) {
                double a, b, c;
                if (PointInTriangle2D(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a, b, c)) {
                    double fi = a * fip + b * fiq + c * fir; // intersection i coordinate
                    int i_interval = int(std::ceil(fi)); // intersection is in (i_interval-1,i_interval]
                    if (i_interval < 0) ++intersection_count(0, j, k); // we enlarge the first interval to include everything to the -x direction
                    else if (i_interval < ni) ++intersection_count(i_interval, j, k);
                    // we ignore intersections that are beyond the +x side of the grid
                }
            }
        }

        // and now we fill in the rest of the distances with fast sweeping
        for (unsigned int pass = 0; pass < 2; ++pass) {
            Sweep(tri, x, phi, closest_tri, origin, dx, +1, +1, +1);
            Sweep(tri, x, phi, closest_tri, origin, dx, -1, -1, -1);
            Sweep(tri, x, phi, closest_tri, origin, dx, +1, +1, -1);
            Sweep(tri, x, phi, closest_tri, origin, dx, -1, -1, +1);
            Sweep(tri, x, phi, closest_tri, origin, dx, +1, -1, +1);
            Sweep(tri, x, phi, closest_tri, origin, dx, -1, +1, -1);
            Sweep(tri, x, phi, closest_tri, origin, dx, +1, -1, -1);
            Sweep(tri, x, phi, closest_tri, origin, dx, -1, +1, +1);
        }

        // then figure out signs (inside/outside) from intersection counts
        for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) {
            int total_count = 0;
            for (int i = 0; i < ni; ++i) {
                total_count += intersection_count(i, j, k);
                if (total_count % 2 == 1) { // if parity of intersections so far is odd,
                    phi(i, j, k) = -phi(i, j, k); // we are inside the mesh
                }
            }
        }
	}

    float GPUSDF::PointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3)
    {
        // first find barycentric coordinates of closest point on infinite plane
        glm::vec3 x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
        float m13 = glm::length(x13), m23 = glm::length(x23), d = dot(x13, x23);
        float invdet = 1.f / max(m13 * m23 - d * d, 1e-30f);
        float a = dot(x13, x03), b = dot(x23, x03);
        // the barycentric coordinates themselves
        float w23 = invdet * (m23 * a - d * b);
        float w31 = invdet * (m13 * b - d * a);
        float w12 = 1 - w23 - w31;
        if (w23 >= 0 && w31 >= 0 && w12 >= 0) {  // if we're inside the triangle
            return glm::distance(x0, w23 * x1 + w31 * x2 + w12 * x3);
        }
        else {        // we have to clamp to one of the edges
            if (w23 > 0) {
                return std::min(PointToSegmentDistance(x0, x1, x2), PointToSegmentDistance(x0, x1, x3));
            }
            else if (w31 > 0) {
                return std::min(PointToSegmentDistance(x0, x1, x2), PointToSegmentDistance(x0, x2, x3));
            }
            else {
                return std::min(PointToSegmentDistance(x0, x1, x3), PointToSegmentDistance(x0, x2, x3));
            }
        }
    }

    float GPUSDF::PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2)
    {
        glm::vec3 dx(x2 - x1);
        double m2 = glm::length(dx);
        // find parameter value of closest point on segment
        float s12 = (float)(dot(x2 - x0, dx) / m2);
        if (s12 < 0) {
            s12 = 0;
        }
        else if (s12 > 1) {
            s12 = 1;
        }

        // and find the distance
        float d = glm::distance(x0, s12 * x1 + (1 - s12) * x2);
        return d;
    }

    bool GPUSDF::PointInTriangle2D(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double& a, double& b, double& c)
    {
        x1 -= x0; x2 -= x0; x3 -= x0;
        y1 -= y0; y2 -= y0; y3 -= y0;
        int signa = Orientation(x2, y2, x3, y3, a);
        if (signa == 0) return false;
        int signb = Orientation(x3, y3, x1, y1, b);
        if (signb != signa) return false;
        int signc = Orientation(x1, y1, x2, y2, c);
        if (signc != signa) return false;
        double sum = a + b + c;
        assert(sum != 0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
        a /= sum;
        b /= sum;
        c /= sum;
        return true;
    }

    int GPUSDF::Orientation(double x1, double y1, double x2, double y2, double& twice_signed_area)
    {
        twice_signed_area = y1 * x2 - x1 * y2;
        if (twice_signed_area > 0) return 1;
        else if (twice_signed_area < 0) return -1;
        else if (y2 > y1) return 1;
        else if (y2 < y1) return -1;
        else if (x1 > x2) return 1;
        else if (x1 < x2) return -1;
        else return 0; // only true when x1==x2 and y1==y2
    }

    void GPUSDF::Sweep(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3f& phi, Array3i& closest_tri, const glm::vec3& origin, float dx, int di, int dj, int dk)
    {
        int i0, i1;
        if (di > 0) { i0 = 1; i1 = phi.ni; }
        else { i0 = phi.ni - 2; i1 = -1; }
        int j0, j1;
        if (dj > 0) { j0 = 1; j1 = phi.nj; }
        else { j0 = phi.nj - 2; j1 = -1; }
        int k0, k1;
        if (dk > 0) { k0 = 1; k1 = phi.nk; }
        else { k0 = phi.nk - 2; k1 = -1; }
        for (int k = k0; k != k1; k += dk) for (int j = j0; j != j1; j += dj) for (int i = i0; i != i1; i += di) {
            glm::vec3 gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i, j, k - dk);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k - dk);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k - dk);
            CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k - dk);
        }
    }

    void GPUSDF::CheckNeighbor(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3f& phi, Array3i& closest_tri, const glm::vec3& gx, int i0, int j0, int k0, int i1, int j1, int k1)
    {
        if (closest_tri(i1, j1, k1) >= 0) {

            auto v = tri[closest_tri(i1, j1, k1)];
            unsigned int p = v.x;
            unsigned int q = v.y;
            unsigned int r = v.z;

            float d = PointToTriangleDistance(gx, x[p], x[q], x[r]);
            if (d < phi(i0, j0, k0)) {
                phi(i0, j0, k0) = d;
                closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
            }
        }
    }
}