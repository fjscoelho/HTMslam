// pointcloud_utils.hpp
//
// Shared helpers to read a sensor_msgs/PointCloud2 into plain (x, y, z)
// doubles. Points with non-finite coordinates are discarded.

#ifndef POINTCLOUD_UTILS_HPP_
#define POINTCLOUD_UTILS_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace pointcloud_utils
{

using Point3d = std::array<double, 3>;

template<typename T>
void copy_points(
  const sensor_msgs::msg::PointCloud2 & msg, std::vector<Point3d> & points)
{
  sensor_msgs::PointCloud2ConstIterator<T> x_it(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<T> y_it(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<T> z_it(msg, "z");

  points.clear();
  points.reserve(static_cast<std::size_t>(msg.width) * msg.height);
  for (; x_it != x_it.end(); ++x_it, ++y_it, ++z_it) {
    const double x = static_cast<double>(*x_it);
    const double y = static_cast<double>(*y_it);
    const double z = static_cast<double>(*z_it);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    points.push_back(Point3d{{x, y, z}});
  }
}

/// Throws std::runtime_error when the cloud cannot be read.
inline void read_points(
  const sensor_msgs::msg::PointCloud2 & msg, std::vector<Point3d> & points)
{
  int x_datatype = -1;
  for (const auto & field : msg.fields) {
    if (field.name == "x") {
      x_datatype = field.datatype;
      break;
    }
  }
  if (x_datatype < 0) {
    throw std::runtime_error("cloud has no 'x' field");
  }

  if (x_datatype == sensor_msgs::msg::PointField::FLOAT32) {
    copy_points<float>(msg, points);
  } else if (x_datatype == sensor_msgs::msg::PointField::FLOAT64) {
    copy_points<double>(msg, points);
  } else {
    throw std::runtime_error("unsupported 'x' field datatype");
  }
}

}  // namespace pointcloud_utils

#endif  // POINTCLOUD_UTILS_HPP_
