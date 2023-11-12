#include "Poseg2o.h"
#include <iostream>

#include "Thirdparty/g2o/g2o/core/factory.h"
#include "Thirdparty/g2o/g2o/stuff/macros.h"

namespace g2o
{
    using namespace std;

    // vertex motion weights
    VertexSDynamicWeight::VertexSDynamicWeight() : BaseVertex<1, double>()
    {}

    bool VertexSDynamicWeight::read(std::istream &is)
    {
        is >> _estimate;
        return true;
    }

    bool VertexSDynamicWeight::write(std::ostream &os) const
    {
        os << _estimate << " ";
        return os.good();
    }

    // binary edge
    bool DynamicWeightEdgeStereoSE3Pose::write(std::ostream &os) const
    {
        for (int i = 0; i <= 3; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    bool DynamicWeightEdgeStereoSE3Pose::read(std::istream &is)
    {
        for (int i = 0; i <= 3; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++)
            { // Matrix<double, D, D>
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    Vector3d DynamicWeightEdgeStereoSE3Pose::cam_project(const Vector3d &trans_xyz, const float &bf) const
    {
        const float invz = 1.0f / trans_xyz[2];
        Vector3d res;
        res[0] = trans_xyz[0] * invz * fx + cx;
        res[1] = trans_xyz[1] * invz * fy + cy;
        res[2] = res[0] - bf * invz;
        return res;
    }

    void DynamicWeightEdgeStereoSE3Pose::linearizeOplus()
    {
        VertexSDynamicWeight *vj = static_cast<VertexSDynamicWeight *>(_vertices[1]);
        double wi = vj->estimate();

        VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        SE3Quat T(vi->estimate());
        Vector3d xyz_trans = T.map(Xw); // 经过位姿变换之后的3D点

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0f / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = wi * (x * y * invz_2 * fx);
        _jacobianOplusXi(0, 1) = wi * (-(1 + (x * x * invz_2)) * fx);
        _jacobianOplusXi(0, 2) = wi * (y * invz * fx);
        _jacobianOplusXi(0, 3) = wi * (-invz * fx);
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = wi * (x * invz_2 * fx);

        _jacobianOplusXi(1, 0) = wi * ((1 + y * y * invz_2) * fy);
        _jacobianOplusXi(1, 1) = wi * (-x * y * invz_2 * fy);
        _jacobianOplusXi(1, 2) = wi * (-x * invz * fy);
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = wi * (-invz * fy);
        _jacobianOplusXi(1, 5) = wi * (y * invz_2 * fy);

        _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - wi * (bf * y * invz_2);
        _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + wi * (bf * x * invz_2);
        _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
        _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - wi * (bf * invz_2);

        Vector3d usobs(_measurement); // 误差相对权重的导数
        _jacobianOplusXj(0, 0) = usobs[0] - (x * invz * fx + cx);
        _jacobianOplusXj(1, 0) = usobs[1] - (y * invz * fy + cy);
        _jacobianOplusXj(2, 0) = usobs[2] - (x * invz * fx + cx - bf * invz);
    }

    // edge
    bool WeightEdgeStereoSE3Pose::write(std::ostream &os) const
    {
        for (int i = 0; i <= 3; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    bool WeightEdgeStereoSE3Pose::read(std::istream &is)
    {
        for (int i = 0; i <= 3; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++)
            { // Matrix<double, D, D>
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    Vector3d WeightEdgeStereoSE3Pose::cam_project(const Vector3d &trans_xyz, const float &bf) const
    {
        const float invz = 1.0f / trans_xyz[2];
        Vector3d res;
        res[0] = trans_xyz[0] * invz * fx + cx;
        res[1] = trans_xyz[1] * invz * fy + cy;
        res[2] = res[0] - bf * invz;
        return res;
    }

    void WeightEdgeStereoSE3Pose::linearizeOplus()
    {
        VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        SE3Quat T(vi->estimate());
        Vector3d xyz_trans = T.map(Xw); // 经过位姿变换之后的3D点

        double wi = weight;
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = wi * (x * y * invz_2 * fx);
        _jacobianOplusXi(0, 1) = -wi * (1 + (x * x * invz_2)) * fx;
        _jacobianOplusXi(0, 2) = wi * (y * invz * fx);
        _jacobianOplusXi(0, 3) = wi * (-invz * fx);
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = wi * (x * invz_2 * fx);

        _jacobianOplusXi(1, 0) = wi * ((1 + y * y * invz_2) * fy);
        _jacobianOplusXi(1, 1) = wi * (-x * y * invz_2 * fy);
        _jacobianOplusXi(1, 2) = wi * (-x * invz * fy);
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = wi * (-invz * fy);
        _jacobianOplusXi(1, 5) = wi * (y * invz_2 * fy);

        _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - wi * (bf * y * invz_2);
        _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + wi * (bf * x * invz_2);
        _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
        _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - wi * (bf * invz_2);
    }

    // Edge local BA
    WeightEdgeStereoSE3ProjectXYZ::WeightEdgeStereoSE3ProjectXYZ() : BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>()
    {
    }

    bool WeightEdgeStereoSE3ProjectXYZ::write(std::ostream &os) const
    {
        for (int i = 0; i <= 3; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    bool WeightEdgeStereoSE3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i <= 3; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    Vector3d WeightEdgeStereoSE3ProjectXYZ::cam_project(const Vector3d &trans_xyz, const float &bf) const
    {
        const float invz = 1.0f / trans_xyz[2];
        Vector3d res;
        res[0] = trans_xyz[0] * invz * fx + cx;
        res[1] = trans_xyz[1] * invz * fy + cy;
        res[2] = res[0] - bf * invz;
        return res;
    }

    void WeightEdgeStereoSE3ProjectXYZ::linearizeOplus()
    {
        VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3Quat T(vj->estimate());
        VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
        Vector3d xyz = vi->estimate();
        Vector3d xyz_trans = T.map(xyz);

        const Matrix3d R = T.rotation().toRotationMatrix();

        double wi = weight;
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z * z;

        _jacobianOplusXi(0, 0) = wi * (-fx * R(0, 0) / z + fx * x * R(2, 0) / z_2);
        _jacobianOplusXi(0, 1) = wi * (-fx * R(0, 1) / z + fx * x * R(2, 1) / z_2);
        _jacobianOplusXi(0, 2) = wi * (-fx * R(0, 2) / z + fx * x * R(2, 2) / z_2);

        _jacobianOplusXi(1, 0) = wi * (-fy * R(1, 0) / z + fy * y * R(2, 0) / z_2);
        _jacobianOplusXi(1, 1) = wi * (-fy * R(1, 1) / z + fy * y * R(2, 1) / z_2);
        _jacobianOplusXi(1, 2) = wi * (-fy * R(1, 2) / z + fy * y * R(2, 2) / z_2);

        _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - wi * (bf * R(2, 0) / z_2);
        _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - wi * (bf * R(2, 1) / z_2);
        _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - wi * (bf * R(2, 2) / z_2);

        _jacobianOplusXj(0, 0) = wi * x * y / z_2 * fx;
        _jacobianOplusXj(0, 1) = -wi * (1 + (x * x / z_2)) * fx;
        _jacobianOplusXj(0, 2) = wi * y / z * fx;
        _jacobianOplusXj(0, 3) = -wi / z * fx;
        _jacobianOplusXj(0, 4) = 0;
        _jacobianOplusXj(0, 5) = wi * x / z_2 * fx;

        _jacobianOplusXj(1, 0) = wi * (1 + y * y / z_2) * fy;
        _jacobianOplusXj(1, 1) = -wi * x * y / z_2 * fy;
        _jacobianOplusXj(1, 2) = -wi * x / z * fy;
        _jacobianOplusXj(1, 3) = 0;
        _jacobianOplusXj(1, 4) = -wi / z * fy;
        _jacobianOplusXj(1, 5) = wi * y / z_2 * fy;

        _jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - wi * bf * y / z_2;
        _jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) + wi * bf * x / z_2;
        _jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2);
        _jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3);
        _jacobianOplusXj(2, 4) = 0;
        _jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - wi * bf / z_2;
    }
}