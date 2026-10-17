#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32_multi_array.hpp>
#include <std_srvs/srv/empty.hpp>
#include <fstream>
#include <json/json.h>
#include <topological_msgs/msg/sdr_stamped.hpp>
#include <topological_msgs/msg/view_template.hpp>
#include <roaring/roaring.hh>
#include <memory>
#include <iostream>
#include <chrono>
#include <vector>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "local_view_match_with_intervals.hpp"
#include "mode_manager/mode_globals.h"
#include <std_msgs/msg/string.hpp>

/**
 * @brief SpatialViewCellsNode - Visual place recognition and loop closure detection
 * 
 * This ROS2 node processes Sparse Distributed Representations (SDR) from visual features
 * to recognize unique places (visual templates) and detect loop closures. It uses an
 * interval-based matching algorithm with Roaring Bitmaps for efficient similarity 
 * computation and memory usage.
 * 
 * Subscriptions:
 *   - <topic_root>/sdr: Receives SDR representations from visual features
 * 
 * Publications:
 *   - <topic_root>/LocalView/Template: Publishes recognized visual template IDs
 * 
 * Parameters:
 *   - topic_root: Base topic namespace for all subscriptions and publications
 *   - theta_alpha: Similarity threshold for extending intervals (default: 384 active bits)
 *   - theta_rho: Maximum duration of an interval in frames (default: 3 frames)
 *   - score_interval: Similarity threshold for loop closure detection (default: 470 matching bits)
 *   - exclude_recent_intervals: Number of recent intervals to exclude from loop closure detection (default: 3)
 */
class SpatialViewCellsNode : public rclcpp::Node {
public:
    explicit SpatialViewCellsNode()
        : Node("spatial_view_cells_node"), counter_(0) {
        
        // Declare and get parameters
        this->declare_parameter("topic_root", "");
        this->declare_parameter<int>("theta_alpha", 384);
        this->declare_parameter<int>("theta_rho", 3);
        this->declare_parameter<int>("score_interval", 470);
        this->declare_parameter<int>("exclude_recent_intervals", 3);
        
        std::string topic_root = this->get_parameter("topic_root").as_string();
        int theta_alpha = this->get_parameter("theta_alpha").as_int();
        int theta_rho = this->get_parameter("theta_rho").as_int();
        int score_interval = this->get_parameter("score_interval").as_int();
        int exclude_recent_intervals = this->get_parameter("exclude_recent_intervals").as_int();
        
        RCLCPP_INFO(this->get_logger(), "SpatialViewCellsNode Parameters:");
        RCLCPP_INFO(this->get_logger(), "  topic_root: %s", topic_root.c_str());
        RCLCPP_INFO(this->get_logger(), "  theta_alpha: %d", theta_alpha);
        RCLCPP_INFO(this->get_logger(), "  theta_rho: %d", theta_rho);
        RCLCPP_INFO(this->get_logger(), "  score_interval: %d", score_interval);
        RCLCPP_INFO(this->get_logger(), "  exclude_recent_intervals: %d", exclude_recent_intervals);
        
        // Initialize LocalViewMatchWithIntervals
        lv_ = std::make_unique<LocalViewMatchWithIntervals>(
            theta_alpha, theta_rho, score_interval, exclude_recent_intervals
        );
        
        // Create publisher for view templates
        pub_vt_ = this->create_publisher<topological_msgs::msg::ViewTemplate>(
            topic_root + "/LocalView/Template", 10);
        
        // Subscribe to SDR
        sub_sdr_ = this->create_subscription<topological_msgs::msg::SdrStamped>(
            topic_root + "/sdr", 10,
            [this](const topological_msgs::msg::SdrStamped::SharedPtr msg) {
                this->sdr_callback(msg);
            });

        // Add mode subscriber to listen for changes in the system mode (mapping/navigation)
        mode_subscriber_ = create_subscription<std_msgs::msg::String>(
        "/system_mode", 10,
        [this](const std_msgs::msg::String::SharedPtr msg) {
            current_mode_ = msg->data;
            // Atualiza a variável global
            ModeGlobals::getInstance().setMode(current_mode_);
            RCLCPP_INFO(get_logger(), "🔀 PoseCell mode changed to: %s", current_mode_.c_str());
            applyModeChange();
        });
        
        // Set up map directory for saving/loading PoseCell state
            std::string states_dir = "./neoslam_exported_states/";
            
            try {
                std::string pkg_share = ament_index_cpp::get_package_share_directory("neoslam");
                // Return to the src directory of the package
                size_t pos = pkg_share.find("/install/");
                if (pos != std::string::npos) {
                    std::string workspace = pkg_share.substr(0, pos);
                    states_dir = workspace + "/src/neoslam/neoslam_exported_states/";
                    RCLCPP_INFO(this->get_logger(), "Map directory set to: %s", states_dir.c_str());
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Error getting package directory: %s", e.what());
                RCLCPP_WARN(this->get_logger(), "Using default map directory: %s", states_dir.c_str());
            }
            
            // Criate the directory if it doesn't exist
            std::string mkdir_cmd = "mkdir -p " + states_dir;
            system(mkdir_cmd.c_str());

            std::string filename = topic_root + "_spatial_view_state.json";
            std::string full_path = states_dir + filename;

        // Services for exporting and importing SpatialView state
        export_sv_service_ = this->create_service<std_srvs::srv::Empty>(
            "/spatial_view/export_state",
            [this, full_path](const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                   std::shared_ptr<std_srvs::srv::Empty::Response> res) {
                (void)req;
                (void)res;
                Json::Value state = lv_->serialize_to_json();
                std::ofstream file(full_path);
                file << state.toStyledString();
                RCLCPP_INFO(this->get_logger(), "SpatialView state exported to %s", full_path.c_str());
            });

        import_sv_service_ = this->create_service<std_srvs::srv::Empty>(
            "/spatial_view/import_state",
            [this, full_path](const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                   std::shared_ptr<std_srvs::srv::Empty::Response> res) {
                (void)req;
                (void)res;
                Json::Value root;
                std::ifstream file(full_path);
                if (!file.is_open()) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to open %s", full_path.c_str());
                    return;
                }
                file >> root;
                if (lv_->deserialize_from_json(root)) {
                    RCLCPP_INFO(this->get_logger(), "SpatialView state imported");
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Failed to import SpatialView state");
                }
            });

        RCLCPP_INFO(this->get_logger(), "SpatialViewCellsNode initialized successfully!");
    }

private:
    void sdr_callback(const topological_msgs::msg::SdrStamped::SharedPtr msg) {
        auto callback_start = std::chrono::high_resolution_clock::now();
        
        const std::vector<uint32_t>& activeCellsSparse = msg->sdr;
        
        // Create Roaring Bitmap from active cell indices
        Roaring sdr_sparse_bitmap;
        for (const auto& cell_idx : activeCellsSparse) {
            sdr_sparse_bitmap.add(cell_idx);
        }
        
        RCLCPP_DEBUG(this->get_logger(), 
                     "Created SDR bitmap with %lu active cells", 
                     sdr_sparse_bitmap.cardinality());
        
        // Call on_image_map with the Roaring bitmap
        auto [interval_map, template_id] = lv_->on_image_map(sdr_sparse_bitmap, counter_);
        
        RCLCPP_DEBUG(this->get_logger(), "template_id: %d", template_id);
        
        // Create and publish ViewTemplate message
        auto vt_msg = std::make_shared<topological_msgs::msg::ViewTemplate>();
        vt_msg->header.stamp = msg->header.stamp; // Use the same timestamp as the image message to publish futher in the experience map
        vt_msg->current_id = template_id;
        // vt_msg->relative_rad = lv_->get_relative_rad();
        
        pub_vt_->publish(*vt_msg);
        
        counter_++;
        
        auto callback_end = std::chrono::high_resolution_clock::now();
        auto callback_duration = std::chrono::duration_cast<std::chrono::microseconds>(callback_end - callback_start);
        
        RCLCPP_DEBUG(this->get_logger(), 
                     "  [SpatialViewCells] Total callback time: %.2f ms", 
                     callback_duration.count() / 1000.0);
    }

    void applyModeChange()
    {
        if (ModeGlobals::getInstance().isMappingMode()) {
        RCLCPP_INFO(get_logger(), "🗺️ SpatialViewCells: MAPPING mode activated");
        // Habilita criação de experiências
        } else if (ModeGlobals::getInstance().isNavigationMode()) {
        RCLCPP_INFO(get_logger(), "🧭 SpatialViewCells: NAVIGATION mode activated");
        // Desabilita criação de experiências
        }
    }
    
    // LocalViewMatchWithIntervals instance
    std::unique_ptr<LocalViewMatchWithIntervals> lv_;
    
    int counter_;
    
    // ROS2 publishers and subscribers
    rclcpp::Publisher<topological_msgs::msg::ViewTemplate>::SharedPtr pub_vt_;
    rclcpp::Subscription<topological_msgs::msg::SdrStamped>::SharedPtr sub_sdr_;
    std::string current_mode_ = "mapping";
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_subscriber_;

    // Services for exporting and importing state
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_sv_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr import_sv_service_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<SpatialViewCellsNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
