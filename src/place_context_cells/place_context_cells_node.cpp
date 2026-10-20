// place_context_cells_node.cpp
//
// RatSLAM-inspired local view cells for the point-cloud place-recognition
// pipeline. Each observation (point cloud + PlaceContext) is compared against
// the stored PlaceContextTemplates by SDR overlap. When the best overlap
// reaches `match_threshold`, the matched template id and the relative yaw
// between the current cloud and the stored one are published. Otherwise a new
// PlaceContextTemplate is created and its id is published.
//
//   sensor_msgs/PointCloud2 ──────────────┐
//                                         ├──► SDR overlap + stratified histogram ──► PlaceContextMatch
//   topological_msgs/PlaceContext ────────┘
//
// The relative yaw follows `yaw_estimator.py`: an angular histogram is
// computed per radial ring (quantile boundaries), the ring histograms are
// cross-correlated and the peak of the weighted sum gives the yaw. Odometry
// disambiguation is intentionally left to the experience map.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rmw/qos_profiles.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <topological_msgs/msg/place_context.hpp>
#include <topological_msgs/msg/place_context_match.hpp>

#include "pointcloud_utils.hpp"

namespace
{

using Point3d = pointcloud_utils::Point3d;

// ============================================================
// PlaceContextTemplate (RatSLAM VisualTemplate analogue)
// ============================================================
struct PlaceContextTemplate
{
  std::uint32_t id = 0;
  std::vector<std::uint32_t> sdr;
  std::vector<std::uint32_t> winner_cells;
  // n_rings * n_bins, ring-major. Stored as float: half the memory of double
  // and well above the precision needed for the yaw correlation.
  std::vector<float> histogram;
};

// ============================================================
// Stratified angular histogram (yaw_estimator.angular_histogram_stratified)
// ============================================================

/// Linear-interpolation percentiles, matching numpy.percentile's default.
std::vector<double> percentile_linear(
  std::vector<double> values, const std::vector<double> & quantiles)
{
  std::vector<double> out;
  out.reserve(quantiles.size());
  if (values.empty()) {
    return std::vector<double>(quantiles.size(), 0.0);
  }
  std::sort(values.begin(), values.end());
  const double n = static_cast<double>(values.size());
  for (const double q : quantiles) {
    double pos = (q / 100.0) * (n - 1.0);
    pos = std::clamp(pos, 0.0, n - 1.0);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    out.push_back(values[lo] + frac * (values[hi] - values[lo]));
  }
  return out;
}

std::vector<float> compute_stratified_histogram(
  const std::vector<Point3d> & points, int n_bins, int n_rings, bool use_centroid)
{
  const std::size_t n = points.size();
  std::vector<double> theta(n);
  std::vector<double> radius(n);

  double anchor_x = 0.0;
  double anchor_y = 0.0;
  if (use_centroid) {
    for (const auto & p : points) {
      anchor_x += p[0];
      anchor_y += p[1];
    }
    anchor_x /= static_cast<double>(n);
    anchor_y /= static_cast<double>(n);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const double dx = points[i][0] - anchor_x;
    const double dy = points[i][1] - anchor_y;
    theta[i] = std::atan2(dy, dx);
    radius[i] = std::hypot(dx, dy);
  }

  // Quantile-based ring boundaries: each ring holds ~the same point count.
  std::vector<double> quantiles(static_cast<std::size_t>(n_rings) + 1);
  for (int k = 0; k <= n_rings; ++k) {
    quantiles[static_cast<std::size_t>(k)] = 100.0 * static_cast<double>(k) / n_rings;
  }
  const std::vector<double> bounds = percentile_linear(radius, quantiles);

  std::vector<double> hist(static_cast<std::size_t>(n_rings) * n_bins, 0.0);
  const double bin_width = 2.0 * M_PI / static_cast<double>(n_bins);
  for (int k = 0; k < n_rings; ++k) {
    const double r_lo = bounds[static_cast<std::size_t>(k)];
    const double r_hi = bounds[static_cast<std::size_t>(k) + 1];
    const bool last_ring = (k == n_rings - 1);
    for (std::size_t i = 0; i < n; ++i) {
      const double r = radius[i];
      if (r < r_lo || (last_ring ? r > r_hi : r >= r_hi)) {
        continue;
      }
      const int bin = std::clamp(
        static_cast<int>((theta[i] + M_PI) / bin_width), 0, n_bins - 1);
      hist[static_cast<std::size_t>(k) * n_bins + bin] += 1.0;
    }
  }

  // Normalize by the total point count (each row sums to its ring fraction)
  // and store as float.
  std::vector<float> result(hist.size());
  for (std::size_t i = 0; i < hist.size(); ++i) {
    result[i] = static_cast<float>(hist[i] / static_cast<double>(n));
  }
  return result;
}

// ============================================================
// Yaw from stratified histograms
// (yaw_estimator.estimate_yaw_from_stratified_histograms)
// ============================================================
double estimate_yaw_from_stratified_histograms(
  const std::vector<float> & hist_a, const std::vector<float> & hist_b,
  int n_rings, int n_bins, bool refine_peak)
{
  // Ring weights: outer rings get more weight, as in the Python pipeline.
  std::vector<double> weights(static_cast<std::size_t>(n_rings));
  double weight_sum = 0.0;
  for (int k = 0; k < n_rings; ++k) {
    weights[static_cast<std::size_t>(k)] =
      (n_rings > 1) ? 1.0 + static_cast<double>(k) / (n_rings - 1) : 1.0;
    weight_sum += weights[static_cast<std::size_t>(k)];
  }
  for (double & w : weights) {
    w /= weight_sum;
  }

  // Circular cross-correlation, matching numpy's irfft(conj(fft(a)) * fft(b)):
  // corr[m] = sum_n a[n] * b[(n + m) mod N].
  std::vector<double> combined(static_cast<std::size_t>(n_bins), 0.0);
  for (int k = 0; k < n_rings; ++k) {
    const float * a = &hist_a[static_cast<std::size_t>(k) * n_bins];
    const float * b = &hist_b[static_cast<std::size_t>(k) * n_bins];
    for (int m = 0; m < n_bins; ++m) {
      double acc = 0.0;
      for (int i = 0; i < n_bins; ++i) {
        int idx = i + m;
        if (idx >= n_bins) {
          idx -= n_bins;
        }
        acc += a[i] * b[idx];
      }
      combined[static_cast<std::size_t>(m)] += weights[static_cast<std::size_t>(k)] * acc;
    }
  }

  int peak_idx = 0;
  for (int m = 1; m < n_bins; ++m) {
    if (combined[static_cast<std::size_t>(m)] > combined[static_cast<std::size_t>(peak_idx)]) {
      peak_idx = m;
    }
  }

  double peak_pos = static_cast<double>(peak_idx);
  if (refine_peak && n_bins >= 3) {
    const double peak = combined[static_cast<std::size_t>(peak_idx)];
    const double left = combined[static_cast<std::size_t>((peak_idx - 1 + n_bins) % n_bins)];
    const double right = combined[static_cast<std::size_t>((peak_idx + 1) % n_bins)];
    const double denom = 2.0 * peak - left - right;
    const double delta = (std::abs(denom) > 1e-12) ?
      std::clamp(0.5 * (right - left) / denom, -1.0, 1.0) : 0.0;
    peak_pos = std::fmod(static_cast<double>(peak_idx) + delta + n_bins, n_bins);
  }

  double yaw = 2.0 * M_PI * peak_pos / static_cast<double>(n_bins);
  yaw = std::fmod(yaw + M_PI, 2.0 * M_PI);
  if (yaw < 0.0) {
    yaw += 2.0 * M_PI;
  }
  return yaw - M_PI;
}

// ============================================================
// Sparse SDR overlap
// ============================================================
std::size_t overlap_count(
  const std::vector<std::uint32_t> & a, const std::vector<std::uint32_t> & b)
{
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t count = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) {
      ++i;
    } else if (b[j] < a[i]) {
      ++j;
    } else {
      ++count;
      ++i;
      ++j;
    }
  }
  return count;
}

double elapsed_ms(
  const std::chrono::steady_clock::time_point & start,
  const std::chrono::steady_clock::time_point & end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================
// Node
// ============================================================
class PlaceContextCellsNode : public rclcpp::Node
{
public:
  using PointCloudMsg = sensor_msgs::msg::PointCloud2;
  using PlaceContextMsg = topological_msgs::msg::PlaceContext;
  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<PointCloudMsg, PlaceContextMsg>;

  PlaceContextCellsNode()
  : Node("place_context_cells_node")
  {
    topic_root_ = this->declare_parameter("topic_root", std::string(""));
    const std::string pointcloud_topic =
      this->declare_parameter("pointcloud_topic", std::string(""));
    const std::string place_context_topic =
      this->declare_parameter("place_context_topic", std::string(""));
    match_threshold_ = this->declare_parameter("match_threshold", 0.85);
    hist_n_bins_ = this->declare_parameter("hist_n_bins", 360);
    hist_n_rings_ = this->declare_parameter("hist_n_rings", 10);
    hist_use_centroid_ = this->declare_parameter("hist_use_centroid", true);
    hist_refine_peak_ = this->declare_parameter("hist_refine_peak", false);
    min_points_ = this->declare_parameter("min_points", 100);
    sync_max_interval_ = this->declare_parameter("sync_max_interval", 0.5);

    if (hist_n_bins_ <= 0 || hist_n_rings_ <= 0) {
      throw std::runtime_error("hist_n_bins and hist_n_rings must be positive");
    }
    if (match_threshold_ < 0.0 || match_threshold_ > 1.0) {
      throw std::runtime_error("match_threshold must be in [0, 1]");
    }

    const std::string cloud_topic = pointcloud_topic.empty() ?
      topic_root_ + "/pointcloud" : pointcloud_topic;
    const std::string context_topic = place_context_topic.empty() ?
      topic_root_ + "/place_context" : place_context_topic;

    publisher_ = this->create_publisher<topological_msgs::msg::PlaceContextMatch>(
      topic_root_ + "/place_context_match", 10);

    cloud_sub_.subscribe(this, cloud_topic, rmw_qos_profile_sensor_data);
    rclcpp::QoS context_qos(10);
    context_sub_.subscribe(this, context_topic, context_qos.get_rmw_qos_profile());

    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), cloud_sub_, context_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_max_interval_));
    sync_->registerCallback(
      std::bind(
        &PlaceContextCellsNode::synced_callback, this,
        std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(
      this->get_logger(),
      "PlaceContextCellsNode ready: threshold=%.2f, histogram=%dx%d",
      match_threshold_, hist_n_rings_, hist_n_bins_);
    RCLCPP_INFO(
      this->get_logger(), "Subscribing to '%s' and '%s', publishing '%s'",
      cloud_topic.c_str(), context_topic.c_str(),
      (topic_root_ + "/place_context_match").c_str());
  }

private:
  /// Bytes actually allocated by the template database (vectors + structs).
  std::size_t database_bytes() const
  {
    std::size_t total = 0;
    for (const auto & t : templates_) {
      total += sizeof(PlaceContextTemplate);
      total += t.sdr.capacity() * sizeof(std::uint32_t);
      total += t.winner_cells.capacity() * sizeof(std::uint32_t);
      total += t.histogram.capacity() * sizeof(float);
    }
    return total;
  }

  void synced_callback(
    const PointCloudMsg::ConstSharedPtr & cloud,
    const PlaceContextMsg::ConstSharedPtr & context)
  {
    const auto start = std::chrono::steady_clock::now();

    std::vector<Point3d> points;
    try {
      pointcloud_utils::read_points(*cloud, points);
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Cannot read point cloud: %s", e.what());
      return;
    }

    if (points.size() < static_cast<std::size_t>(min_points_)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Skipping cloud with too few finite points (%zu < %d)",
        points.size(), min_points_);
      return;
    }

    if (context->sdr.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Skipping PlaceContext with an empty SDR");
      return;
    }

    const std::vector<float> histogram = compute_stratified_histogram(
      points, hist_n_bins_, hist_n_rings_, hist_use_centroid_);
    const auto histogram_done = std::chrono::steady_clock::now();

    // ---- Match against every stored template (SDR overlap) ----
    std::size_t best_overlap = 0;
    int best_idx = -1;
    for (std::size_t i = 0; i < templates_.size(); ++i) {
      const std::size_t shared = overlap_count(context->sdr, templates_[i].sdr);
      if (shared > best_overlap) {
        best_overlap = shared;
        best_idx = static_cast<int>(i);
      }
    }
    const double ratio =
      static_cast<double>(best_overlap) / static_cast<double>(context->sdr.size());
    const auto overlap_done = std::chrono::steady_clock::now();

    topological_msgs::msg::PlaceContextMatch out;
    out.header = context->header;
    out.overlap = ratio;

    double yaw_ms = 0.0;
    if (best_idx >= 0 && ratio >= match_threshold_) {
      const PlaceContextTemplate & matched = templates_[static_cast<std::size_t>(best_idx)];
      const auto yaw_start = std::chrono::steady_clock::now();
      out.relative_rad = estimate_yaw_from_stratified_histograms(
        matched.histogram, histogram, hist_n_rings_, hist_n_bins_, hist_refine_peak_);
      yaw_ms = elapsed_ms(yaw_start, std::chrono::steady_clock::now());

      out.matched = true;
      out.context_cell_id = matched.id;

      RCLCPP_DEBUG(
        this->get_logger(),
        "Matched template %u (overlap=%.3f, yaw=%.2f deg)",
        matched.id, ratio, out.relative_rad * 180.0 / M_PI);
    } else {
      PlaceContextTemplate created;
      created.id = next_id_++;
      created.sdr = context->sdr;
      created.winner_cells = context->winner_cells;
      created.histogram = histogram;
      templates_.push_back(std::move(created));

      out.matched = false;
      out.context_cell_id = templates_.back().id;
      out.relative_rad = 0.0;

      RCLCPP_INFO(
        this->get_logger(),
        "New PlaceContextTemplate %u (best overlap %.3f, templates=%zu)",
        out.context_cell_id, ratio, templates_.size());
    }

    publisher_->publish(out);
    const auto end = std::chrono::steady_clock::now();

    RCLCPP_DEBUG(
      this->get_logger(),
      "Timing: histogram %.2f ms | overlap %.2f ms | yaw %.2f ms | total %.2f ms"
      " | templates=%zu | db=%.2f KB",
      elapsed_ms(start, histogram_done),
      elapsed_ms(histogram_done, overlap_done),
      yaw_ms,
      elapsed_ms(start, end),
      templates_.size(),
      static_cast<double>(database_bytes()) / 1024.0);
  }

  std::string topic_root_;
  double match_threshold_ = 0.85;
  int hist_n_bins_ = 360;
  int hist_n_rings_ = 10;
  bool hist_use_centroid_ = true;
  bool hist_refine_peak_ = false;
  int min_points_ = 100;
  double sync_max_interval_ = 0.5;

  std::vector<PlaceContextTemplate> templates_;
  std::uint32_t next_id_ = 0;

  rclcpp::Publisher<topological_msgs::msg::PlaceContextMatch>::SharedPtr publisher_;
  message_filters::Subscriber<PointCloudMsg> cloud_sub_;
  message_filters::Subscriber<PlaceContextMsg> context_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<PlaceContextCellsNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
