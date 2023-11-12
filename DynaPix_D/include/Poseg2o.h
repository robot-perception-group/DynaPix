/*
Add VertexSDynamicWeight()    //identify Vertex for dynamic weight
Add EdgeStereoSE3PoseDynamicWeght()  // Edge　for optimize pose and dynamic weight
*/

#ifndef POSEG20_H
#define POSEG20_H

#include "Thirdparty/g2o/g2o/core/optimizable_graph.h"

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include <Eigen/Geometry>

namespace g2o 
{

using namespace Eigen;

class VertexSDynamicWeight : public BaseVertex<1, double>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
	  
    VertexSDynamicWeight();
    
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {//设置初始估计值
      _estimate  = 1.0;
    }

    virtual void oplusImpl(const double* update)//更新
    { 
      _estimate += *update;
    }
};

//观测值维度．类型，连接顶点类型．
// Pose Optimization (Optimize Only Pose)
class  WeightEdgeStereoSE3Pose: public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  WeightEdgeStereoSE3Pose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);     
    Vector3d obs(_measurement);    //_measurement类型为：typedef E Measurement

    Vector3d temp = obs - cam_project(v1->estimate().map(Xw), bf);
    for (int i=0; i<3; i++){
      _error[i] = weight*temp[i];
    }
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }

  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz, const float &bf) const;

  Vector3d Xw;

  double weight;
  double fx, fy, cx, cy, bf;
};

// Local BA
class WeightEdgeStereoSE3ProjectXYZ : public BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  WeightEdgeStereoSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d obs(_measurement);

    Vector3d temp = obs - cam_project(v1->estimate().map(v2->estimate()),bf);
    for (int i=0; i<3; i++){
      _error[i] = weight*temp[i];
      }
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }

  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz, const float &bf) const;

  double weight;

  double fx, fy, cx, cy, bf;
};


// Binary Edge for Pose Optimization
class  DynamicWeightEdgeStereoSE3Pose: public  BaseBinaryEdge<3, Vector3d, VertexSE3Expmap, VertexSDynamicWeight>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DynamicWeightEdgeStereoSE3Pose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    const VertexSDynamicWeight* v2 = static_cast<const VertexSDynamicWeight*>(_vertices[1]);     
    Vector3d obs(_measurement);    //_measurement类型为：typedef E Measurement
     //_error类型为：Matrix<double, D, 1> ErrorVector
    double w = v2->estimate();
    
    Vector3d temp = obs - cam_project(v1->estimate().map(Xw), bf);
    for (int i=0; i<3; i++){
      _error[i] = w*temp[i];
    }
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }

  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz, const float &bf) const;

  Vector3d Xw;
  double fx, fy, cx, cy, bf;
};
}

#endif//POSEBIG20_H