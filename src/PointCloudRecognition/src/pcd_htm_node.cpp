// pcd_htm_node.cpp
//
// ROS 2 wrapper around the place-recognition blocks of PointCloudRecognition:
//
//   sensor_msgs/PointCloud2 ──► PlaceDescriptor ──► SDRPlaceEncoder ──► SDR
//                                                                        │
//                                                             HTM TemporalMemory
//                                                                        │
//                                                        topological_msgs/PlaceContext
//
// The node reproduces, in C++, the behaviour of `place_encoder.py` and
// `place_context.py`:
//   * the descriptor is computed in the sensor frame (origin = sensor) and is
//     yaw-invariant;
//   * the encoder configuration (ranges, resolutions, sizes and active bits)
//     is loaded from `encoder_config.json`;
//   * the Temporal Memory has one mini-column per SDR bit, so the encoder SDR
//     is fed directly as the active mini-columns (no SpatialPooler).
//
// The TM hyper-parameters are hard-coded to the defaults used by
// `neocortex_node.cpp`.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <topological_msgs/msg/place_context.hpp>

#include "pointcloud_utils.hpp"

#include "htm/algorithms/TemporalMemory.hpp"
#include "htm/encoders/RandomDistributedScalarEncoder.hpp"
#include "htm/types/Sdr.hpp"

namespace
{

// ============================================================
// HTM Temporal Memory defaults (same as neocortex_node.cpp)
// ============================================================
constexpr htm::UInt kCellsPerColumn = 32;
constexpr htm::UInt kActivationThreshold = 4;
constexpr htm::Real kInitialPermanence = 0.55;
constexpr htm::Real kConnectedPermanence = 0.5;
constexpr htm::UInt kMinThreshold = 1;
constexpr htm::UInt kMaxNewSynapseCount = 20;
constexpr htm::Real kPermanenceIncrement = 0.01;
constexpr htm::Real kPermanenceDecrement = 0.01;
constexpr htm::Real kPredictedSegmentDecrement = 0.0005;
constexpr htm::UInt kSeed = 42;
constexpr htm::UInt kMaxSegmentsPerCell = 100;
constexpr htm::UInt kMaxSynapsesPerSegment = 100;

constexpr std::size_t kMinPoints = 100;
constexpr int kHistBins = 10;

double elapsed_ms(
  const std::chrono::steady_clock::time_point & start,
  const std::chrono::steady_clock::time_point & end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================
// PlaceDescriptor (equivalent to place_encoder.PlaceDescriptor)
// ============================================================
using DescriptorValues = std::map<std::string, double>;

DescriptorValues compute_descriptor(const std::vector<std::array<double, 3>> & points)
{
  const std::size_t n = points.size();

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto & p : points) {
    mean += Eigen::Vector3d(p[0], p[1], p[2]);
  }
  mean /= static_cast<double>(n);

  // Same as np.cov(pts.T): sample covariance (ddof = 1).
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto & p : points) {
    const Eigen::Vector3d d = Eigen::Vector3d(p[0], p[1], p[2]) - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<double>(n - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("eigen decomposition of the point covariance failed");
  }
  // Eigen returns eigenvalues in ascending order; np.linalg.eigvalsh is also
  // ascending and the Python code reverses it.
  std::array<double, 3> eigvals{{
    std::max(0.0, solver.eigenvalues()[2]),
    std::max(0.0, solver.eigenvalues()[1]),
    std::max(0.0, solver.eigenvalues()[0])}};

  std::vector<double> z(n);
  std::vector<double> r(n);
  double z_min = points[0][2];
  double z_max = points[0][2];
  double r_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    z[i] = points[i][2];
    r[i] = std::hypot(points[i][0], points[i][1]);
    z_min = std::min(z_min, z[i]);
    z_max = std::max(z_max, z[i]);
    r_max = std::max(r_max, r[i]);
  }
  auto histogram = [n](const std::vector<double> & values, double lo, double hi) {
      if (hi - lo < 1e-9) {
        lo -= 0.5;
        hi += 0.5;
      }
      std::vector<double> hist(kHistBins, 0.0);
      const double width = (hi - lo) / static_cast<double>(kHistBins);
      for (const double v : values) {
        if (v < lo || v > hi) {
          continue;
        }
        int idx = static_cast<int>((v - lo) / width);
        idx = std::min(idx, kHistBins - 1);
        hist[static_cast<std::size_t>(idx)] += 1.0;
      }
      for (double & h : hist) {
        h /= static_cast<double>(n);
      }
      return hist;
    };

  const std::vector<double> hist_z = histogram(z, z_min, z_max);
  const std::vector<double> hist_r = histogram(r, 0.0, r_max);

  double sum_r = 0.0;
  double sum_z = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum_r += r[i];
    sum_z += z[i];
  }
  const double mean_r = sum_r / static_cast<double>(n);
  const double mean_z = sum_z / static_cast<double>(n);

  double var_r = 0.0;
  double var_z = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    var_r += (r[i] - mean_r) * (r[i] - mean_r);
    var_z += (z[i] - mean_z) * (z[i] - mean_z);
  }
  const double std_r = std::sqrt(var_r / static_cast<double>(n));
  const double std_z = std::sqrt(var_z / static_cast<double>(n));

  DescriptorValues values;
  values["eigval_1"] = eigvals[0];
  values["eigval_2"] = eigvals[1];
  values["eigval_3"] = eigvals[2];
  for (int i = 0; i < kHistBins; ++i) {
    values["hist_z_" + std::to_string(i)] = hist_z[static_cast<std::size_t>(i)];
    values["hist_r_" + std::to_string(i)] = hist_r[static_cast<std::size_t>(i)];
  }
  values["height"] = z_max - z_min;
  values["density"] = std::log1p(static_cast<double>(n));
  values["volume"] = (4.0 / 3.0) * M_PI *
    std::sqrt(eigvals[0] * eigvals[1] * eigvals[2] + 1e-12);
  values["mean_radius"] = mean_r;
  values["std_radius"] = std_r;
  values["mean_height"] = mean_z;
  values["std_height"] = std_z;
  return values;
}

// ============================================================
// SDRPlaceEncoder (equivalent to place_encoder.SDRPlaceEncoder)
// ============================================================
class SdrPlaceEncoder
{
public:
  explicit SdrPlaceEncoder(const std::string & config_path)
  {
    std::ifstream stream(config_path);
    if (!stream.is_open()) {
      throw std::runtime_error("cannot open encoder config: " + config_path);
    }
    nlohmann::json config;
    stream >> config;

    const auto & ranges = config.at("feature_ranges");
    const auto & resolutions = config.at("feature_resolutions");
    const std::size_t n_features = ranges.size();
    if (resolutions.size() != n_features) {
      throw std::runtime_error(
              "encoder config: feature_resolutions and feature_ranges differ in size");
    }

    const auto size_at = [&config](std::size_t i) -> htm::UInt {
        const auto & sizes = config.at("feature_sizes");
        return sizes.is_array() ? sizes.at(i).get<htm::UInt>() : sizes.get<htm::UInt>();
      };
    const auto active_bits_at = [&config](std::size_t i) -> htm::UInt {
        const auto & active = config.at("active_bits");
        return active.is_array() ? active.at(i).get<htm::UInt>() : active.get<htm::UInt>();
      };
    const htm::UInt seed = config.value("seed", 42);

    std::vector<std::string> names;
    if (config.contains("feature_names")) {
      names = config.at("feature_names").get<std::vector<std::string>>();
      if (names.size() != n_features) {
        throw std::runtime_error(
                "encoder config: feature_names and feature_ranges differ in size");
      }
    }

    for (std::size_t i = 0; i < n_features; ++i) {
      Feature feature;
      feature.name = names.empty() ? ("feature_" + std::to_string(i)) : names[i];
      feature.range_lo = ranges.at(i).at(0).get<double>();
      feature.range_hi = ranges.at(i).at(1).get<double>();
      feature.resolution = resolutions.at(i).get<double>();
      feature.size = size_at(i);
      feature.active_bits = active_bits_at(i);

      const double n_buckets = (feature.range_hi - feature.range_lo) /
        std::max(feature.resolution, 1e-12);
      if (n_buckets > static_cast<double>(feature.size) / 2.0) {
        throw std::runtime_error(
                "encoder config: too many buckets (" + std::to_string(n_buckets) +
                ") for feature '" + feature.name + "' with size " +
                std::to_string(feature.size));
      }
      if (feature.active_bits >= feature.size) {
        throw std::runtime_error(
                "encoder config: active_bits >= size for feature '" + feature.name + "'");
      }

      htm::RDSE_Parameters params;
      params.size = feature.size;
      params.activeBits = feature.active_bits;
      params.resolution = feature.resolution;
      params.seed = seed;

      // RDSE validates its parameters with a randomized collision check for
      // sizes below 1000 bits. That check can fail sporadically even for valid
      // parameters, so retry a few times before reporting the error.
      std::unique_ptr<htm::RandomDistributedScalarEncoder> encoder;
      for (int attempt = 0; attempt < 10 && !encoder; ++attempt) {
        try {
          encoder = std::make_unique<htm::RandomDistributedScalarEncoder>(params);
        } catch (const std::exception & e) {
          if (attempt == 9) {
            throw std::runtime_error(
                    "cannot build RDSE encoder for feature '" + feature.name + "': " + e.what());
          }
        }
      }
      encoders_.push_back(std::move(encoder));
      features_.push_back(feature);
      total_size_ += feature.size;
    }

    if (total_size_ == 0) {
      throw std::runtime_error("encoder config: SDR size is zero");
    }
  }

  std::uint32_t total_size() const {return total_size_;}

  // Returns the sorted sparse indices of the concatenated SDR.
  const std::vector<htm::UInt> & encode(const DescriptorValues & values)
  {
    active_.clear();
    htm::UInt offset = 0;
    for (std::size_t i = 0; i < features_.size(); ++i) {
      const Feature & feature = features_[i];
      const auto it = values.find(feature.name);
      if (it == values.end()) {
        throw std::runtime_error("descriptor has no feature named '" + feature.name + "'");
      }
      htm::SDR out(std::vector<htm::UInt>{feature.size});
      encoders_[i]->encode(it->second, out);
      for (const htm::UInt idx : out.getSparse()) {
        active_.push_back(offset + idx);
      }
      offset += feature.size;
    }
    std::sort(active_.begin(), active_.end());
    return active_;
  }

private:
  struct Feature
  {
    std::string name;
    double range_lo = 0.0;
    double range_hi = 0.0;
    double resolution = 0.0;
    htm::UInt size = 0;
    htm::UInt active_bits = 0;
  };

  std::vector<Feature> features_;
  std::vector<std::unique_ptr<htm::RandomDistributedScalarEncoder>> encoders_;
  std::vector<htm::UInt> active_;
  std::uint32_t total_size_ = 0;
};

// ============================================================
// Node
// ============================================================
class PcdHtmNode : public rclcpp::Node
{
public:
  PcdHtmNode()
  : Node("pcd_htm_node")
  {
    topic_root_ = this->declare_parameter("topic_root", std::string(""));
    const std::string pointcloud_topic =
      this->declare_parameter("pointcloud_topic", std::string(""));
    std::string encoder_config =
      this->declare_parameter("encoder_config", std::string(""));

    if (encoder_config.empty()) {
      encoder_config = ament_index_cpp::get_package_share_directory("htmslam") +
        "/config/encoder_config.json";
    }

    encoder_ = std::make_unique<SdrPlaceEncoder>(encoder_config);
    const std::uint32_t sdr_size = encoder_->total_size();
    RCLCPP_INFO(
      this->get_logger(), "Encoder config loaded from '%s'", encoder_config.c_str());

    tm_ = std::make_unique<htm::TemporalMemory>(
      std::vector<htm::UInt>{static_cast<htm::UInt>(sdr_size)},
      kCellsPerColumn,
      kActivationThreshold,
      kInitialPermanence,
      kConnectedPermanence,
      kMinThreshold,
      kMaxNewSynapseCount,
      kPermanenceIncrement,
      kPermanenceDecrement,
      kPredictedSegmentDecrement,
      kSeed,
      kMaxSegmentsPerCell,
      kMaxSynapsesPerSegment);

    input_sdr_ = std::make_unique<htm::SDR>(
      std::vector<htm::UInt>{static_cast<htm::UInt>(sdr_size)});
    winner_cells_sdr_ = std::make_unique<htm::SDR>(
      std::vector<htm::UInt>{static_cast<htm::UInt>(tm_->numberOfCells())});

    const std::string cloud_topic = pointcloud_topic.empty() ?
      topic_root_ + "/pointcloud" : pointcloud_topic;

    publisher_ = this->create_publisher<topological_msgs::msg::PlaceContext>(
      topic_root_ + "/place_context", 10);

    subscription_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pointcloud_callback(msg);
      });

    RCLCPP_INFO(
      this->get_logger(),
      "PcdHtmNode ready: encoder SDR=%u bits, TM columns=%u, cells=%zu",
      sdr_size, static_cast<unsigned>(tm_->numberOfColumns()), tm_->numberOfCells());
    RCLCPP_INFO(
      this->get_logger(), "Subscribing to '%s', publishing '%s'",
      cloud_topic.c_str(), (topic_root_ + "/place_context").c_str());
  }

private:
  using Point3d = pointcloud_utils::Point3d;

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto callback_start = std::chrono::steady_clock::now();

    std::vector<Point3d> points;
    try {
      pointcloud_utils::read_points(*msg, points);
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Cannot read point cloud: %s", e.what());
      return;
    }

    if (points.size() < kMinPoints) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Skipping cloud with too few finite points (%zu < %zu)",
        points.size(), kMinPoints);
      return;
    }

    try {
      const DescriptorValues descriptor = compute_descriptor(points);
      const std::vector<htm::UInt> & active_bits = encoder_->encode(descriptor);
      const auto sdr_done = std::chrono::steady_clock::now();

      input_bits_.assign(active_bits.begin(), active_bits.end());
      input_sdr_->setSparse(input_bits_);
      tm_->compute(*input_sdr_, /*learn=*/true);
      tm_->getWinnerCells(*winner_cells_sdr_);
      const auto htm_done = std::chrono::steady_clock::now();

      topological_msgs::msg::PlaceContext out;
      out.header = msg->header;
      out.sdr_size = encoder_->total_size();
      out.sdr = active_bits;
      out.num_columns = static_cast<std::uint32_t>(tm_->numberOfColumns());
      out.cells_per_column = static_cast<std::uint32_t>(tm_->getCellsPerColumn());
      out.winner_cells = winner_cells_sdr_->getSparse();
      out.anomaly = tm_->anomaly;
      out.iteration = static_cast<std::uint32_t>(iteration_);

      publisher_->publish(out);
      ++iteration_;

      const auto callback_end = std::chrono::steady_clock::now();

      RCLCPP_DEBUG(
        this->get_logger(),
        "Timing: PCD->SDR %.2f ms | HTM %.2f ms | total %.2f ms",
        elapsed_ms(callback_start, sdr_done),
        elapsed_ms(sdr_done, htm_done),
        elapsed_ms(callback_start, callback_end));

      RCLCPP_DEBUG(
        this->get_logger(),
        "SDR bits=%zu winner cells=%zu anomaly=%.4f",
        out.sdr.size(), out.winner_cells.size(), out.anomaly);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Failed to process cloud: %s", e.what());
    }
  }

  std::string topic_root_;
  std::unique_ptr<SdrPlaceEncoder> encoder_;
  std::unique_ptr<htm::TemporalMemory> tm_;
  std::unique_ptr<htm::SDR> input_sdr_;
  std::vector<htm::UInt> input_bits_;
  std::unique_ptr<htm::SDR> winner_cells_sdr_;
  std::uint64_t iteration_ = 0;

  rclcpp::Publisher<topological_msgs::msg::PlaceContext>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_cloud_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<PcdHtmNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
