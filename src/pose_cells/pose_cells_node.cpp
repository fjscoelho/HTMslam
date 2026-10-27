#include <cmath>
#include <iostream>
#include <memory>
#include <functional>
using namespace std;

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>
#include <fstream>
#include <json/json.h>

#include "posecell_network.h"
#include "mode_manager/mode_globals.h"  // <-- NOVO INCLUDE
#include <topological_msgs/msg/topological_action.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <topological_msgs/msg/place_context_match.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace
{
// Translation below this (metres per odometry step) has no reliable direction,
// so the holonomic heading offset is kept at zero.
constexpr double kMinHolonomicTranslation = 1e-4;
}  // namespace

#if HAVE_IRRLICHT
#include "posecell_scene.h"
PosecellScene *pcs;
bool use_graphics;
#endif

/**
 * @brief PoseCellNode - Continuous attractor network for path integration
 * 
 * This ROS2 node implements a 3D continuous attractor network (pose cells) that
 * performs path integration to estimate the robot's pose. It integrates odometry
 * for dead reckoning and corrects drift using visual template matches. The pose
 * cell activity represents a probability distribution over the robot's x, y, and
 * heading (theta) in a local coordinate frame.
 * 
 * Subscriptions:
 *   - <topic_root>/odom: Receives odometry data for path integration
 *   - <topic_root>/place_context_match: Receives place context cell matches
 *     (id + relative yaw) from place_context_cells_node for correction
 * 
 * Publications:
 *   - <topic_root>/PoseCell/TopologicalAction: Publishes actions for experience map updates
 * 
 * Parameters:
 *   - topic_root: Base topic namespace for all subscriptions and publications
 *   - pc_dim_xy: Dimensions of pose cell network in x and y (default: 21)
 *   - pc_dim_th: Dimensions of pose cell network in theta/heading (default: 36)
 *   - pc_w_e_dim: Size of excitatory weight matrix dimension (default: 7)
 *   - pc_w_i_dim: Size of inhibitory weight matrix dimension (default: 5)
 *   - pc_w_e_var: Variance of excitatory weights (default: 1)
 *   - pc_w_i_var: Variance of inhibitory weights (default: 2)
 *   - pc_global_inhib: Global inhibition constant (default: 0.00002)
 *   - vt_active_decay: Decay rate of visual template activation (default: 1.0)
 *   - pc_vt_inject_energy: Energy injected into pose cells on template match (default: 0.15)
 *   - pc_cell_x_size: Physical size represented by each x/y cell in meters (default: 1.0)
 *   - exp_delta_pc_threshold: Pose change threshold for creating new experiences (default: 2.0)
 *   - pc_vt_restore: Energy restoration factor for visual templates (default: 0.05)
 *   - enable: Enable Irrlicht graphics visualization (default: true)
 *   - posecells_size: Size of visualization window (default: 250)
 *   - media_path: Path to media resources for visualization
 *   - image_file: Image file for visualization background
 *   - holonomic_odometry: Use twist.linear.y as well, decoupling the
 *     translation direction from the heading (default: false)
 */
class PoseCellNode : public rclcpp::Node
{
public:
  PoseCellNode()
    : Node("pose_cells_node")
  {
    // Declare and get parameters
    this->declare_parameter("topic_root", "");
    this->declare_parameter("pc_dim_xy", 21);
    this->declare_parameter("pc_dim_th", 36);
    this->declare_parameter("pc_w_e_dim", 7);
    this->declare_parameter("pc_w_i_dim", 5);
    this->declare_parameter("pc_w_e_var", 1);
    this->declare_parameter("pc_w_i_var", 2);
    this->declare_parameter("pc_global_inhib", 0.00002);
    this->declare_parameter("vt_active_decay", 1.0);
    this->declare_parameter("pc_vt_inject_energy", 0.15);
    this->declare_parameter("pc_cell_x_size", 1.0);
    this->declare_parameter("exp_delta_pc_threshold", 2.0);
    this->declare_parameter("pc_vt_restore", 0.05);
    this->declare_parameter("enable", true);
    this->declare_parameter("posecells_size", 250);
    this->declare_parameter("media_path", "");
    this->declare_parameter("image_file", "");
    this->declare_parameter("holonomic_odometry", false);
    
    std::string topic_root = this->get_parameter("topic_root").as_string();
    holonomic_odometry_ = this->get_parameter("holonomic_odometry").as_bool();
    
    // Log all parameters for debugging
    RCLCPP_INFO(this->get_logger(), "PoseCell Parameters:");
    RCLCPP_INFO(this->get_logger(), "  topic_root: %s", topic_root.c_str());
    RCLCPP_INFO(this->get_logger(), "  pc_dim_xy: %ld", this->get_parameter("pc_dim_xy").as_int());
    RCLCPP_INFO(this->get_logger(), "  pc_dim_th: %ld", this->get_parameter("pc_dim_th").as_int());
    RCLCPP_INFO(this->get_logger(), "  pc_vt_inject_energy: %f", this->get_parameter("pc_vt_inject_energy").as_double());
    RCLCPP_INFO(this->get_logger(), "  pc_cell_x_size: %f", this->get_parameter("pc_cell_x_size").as_double());
    RCLCPP_INFO(this->get_logger(), "  exp_delta_pc_threshold: %f", this->get_parameter("exp_delta_pc_threshold").as_double());
    RCLCPP_INFO(this->get_logger(), "  holonomic_odometry: %s", holonomic_odometry_ ? "true" : "false");
    
    // Create PosecellNetwork with parameters
    pc = new PosecellNetwork(
      this->get_parameter("pc_dim_xy").as_int(),
      this->get_parameter("pc_dim_th").as_int(),
      this->get_parameter("pc_w_e_dim").as_int(),
      this->get_parameter("pc_w_i_dim").as_int(),
      this->get_parameter("pc_w_e_var").as_int(),
      this->get_parameter("pc_w_i_var").as_int(),
      this->get_parameter("pc_global_inhib").as_double(),
      this->get_parameter("vt_active_decay").as_double(),
      this->get_parameter("pc_vt_inject_energy").as_double(),
      this->get_parameter("pc_cell_x_size").as_double(),
      this->get_parameter("exp_delta_pc_threshold").as_double(),
      this->get_parameter("pc_vt_restore").as_double()
    );
    
    pub_pc = this->create_publisher<topological_msgs::msg::TopologicalAction>(
      topic_root + "/PoseCell/TopologicalAction", 10);
    
    sub_odometry = this->create_subscription<nav_msgs::msg::Odometry>(
      topic_root + "/odom", 10,
      std::bind(&PoseCellNode::odo_callback, this, std::placeholders::_1));
    
    sub_place_context = this->create_subscription<topological_msgs::msg::PlaceContextMatch>(
      topic_root + "/place_context_match", 10,
      std::bind(&PoseCellNode::place_context_callback, this, std::placeholders::_1));

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
    
    // Inicializa o modo global com o valor atual (se já existir)
    // Caso contrário, usa "mapping" como padrão
    std::string initial_mode = "mapping";
    try {
      initial_mode = ModeGlobals::getInstance().getMode();
    } catch (...) {
      ModeGlobals::getInstance().setMode("mapping");
    }
    
    RCLCPP_INFO(this->get_logger(), "📝 PoseCell initial mode: %s", initial_mode.c_str());

    // Callback for dynamic parameter updates
    this->add_on_set_parameters_callback(
        std::bind(&PoseCellNode::parametersCallback, this, std::placeholders::_1)
    );

    // Set up map directory for saving/loading PoseCell state
      std::string states_dir = "./htmslam_exported_states/";
      
      try {
          std::string pkg_share = ament_index_cpp::get_package_share_directory("htmslam");
          // Return to the src directory of the package
          size_t pos = pkg_share.find("/install/");
          if (pos != std::string::npos) {
              std::string workspace = pkg_share.substr(0, pos);
              states_dir = workspace + "/src/HTMslam/htmslam_exported_states/";
              RCLCPP_INFO(this->get_logger(), "Map directory set to: %s", states_dir.c_str());
          }
      } catch (const std::exception& e) {
          RCLCPP_WARN(this->get_logger(), "Error getting package directory: %s", e.what());
          RCLCPP_WARN(this->get_logger(), "Using default map directory: %s", states_dir.c_str());
      }
      
      // Criate the directory if it doesn't exist
      std::string mkdir_cmd = "mkdir -p " + states_dir;
      system(mkdir_cmd.c_str());

      std::string filename = topic_root + "_pose_cells_state.json";
      std::string full_path = states_dir + filename;

    // services for exporting and importing PoseCell state
    export_pc_service_ = this->create_service<std_srvs::srv::Empty>(
    "/pose_cells/export_state",
    [this, full_path](const std::shared_ptr<std_srvs::srv::Empty::Request> req,
           std::shared_ptr<std_srvs::srv::Empty::Response> res) {
        (void)req; (void)res;
        Json::Value state = pc->serialize_to_json();
        std::ofstream file(full_path);
        file << state.toStyledString();
        file.close();
        RCLCPP_INFO(this->get_logger(), "PoseCell state exported to %s", full_path.c_str());
    });

    import_pc_service_ = this->create_service<std_srvs::srv::Empty>(
        "/pose_cells/import_state",
        [this, full_path](const std::shared_ptr<std_srvs::srv::Empty::Request> req,
              std::shared_ptr<std_srvs::srv::Empty::Response> res) {
            (void)req; (void)res;
            Json::Value root;
            std::ifstream file(full_path);
            if (!file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open %s", full_path.c_str());
                return;
            }
            file >> root;
            if (pc->deserialize_from_json(root)) {
                RCLCPP_INFO(this->get_logger(), "PoseCell state imported successfully");
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to import PoseCell state");
            }
        });
    
      #ifdef HAVE_IRRLICHT
          use_graphics = this->get_parameter("enable").as_bool();
          if (use_graphics)
          {
            pcs = new PosecellScene(
              this->get_parameter("posecells_size").as_int(),
              this->get_parameter("media_path").as_string(),
              this->get_parameter("image_file").as_string(),
              pc
            );
          }
      #endif
  }
  
  ~PoseCellNode()
  {
    if (pc != nullptr)
      delete pc;
    #ifdef HAVE_IRRLICHT
        if (pcs != nullptr)
          delete pcs;
    #endif
  }

private:
  void odo_callback(const nav_msgs::msg::Odometry::SharedPtr odo)
  {
    RCLCPP_DEBUG(this->get_logger(), "PC:odo_callback v=%f r=%f", 
                 odo->twist.twist.linear.x, odo->twist.twist.angular.z);

    if (prev_time.seconds() > 0)
    {
      double time_diff = (rclcpp::Time(odo->header.stamp) - prev_time).seconds();

      pc_output.src_id = pc->get_current_exp_id();
      if (holonomic_odometry_)
      {
        const double vx = odo->twist.twist.linear.x;
        const double vy = odo->twist.twist.linear.y;
        double speed = std::hypot(vx, vy);
        double dheading = 0.0;
        if (speed * time_diff > kMinHolonomicTranslation)
        {
          dheading = std::atan2(vy, vx);
        }
        else
        {
          speed = 0.0;
        }
        pc->on_odo_holonomic(speed, dheading, odo->twist.twist.angular.z, time_diff);
      }
      else
      {
        pc->on_odo(odo->twist.twist.linear.x, odo->twist.twist.angular.z, time_diff);
      }
      pc_output.action = pc->get_action();
      
      if (pc_output.action != PosecellNetwork::NO_ACTION)
      {
        pc_output.header.stamp = odo->header.stamp;
        pc_output.dest_id = pc->get_current_exp_id();
        pc_output.relative_rad = pc->get_relative_rad();
        pc_output.vt_id = pc->get_current_vt_id();
        pub_pc->publish(pc_output);
        
        RCLCPP_INFO(this->get_logger(), "PC:action_publish action=%d src=%d dest=%d vt_id=%d",
                     pc_output.action, pc_output.src_id, pc_output.dest_id, pc_output.vt_id);
      }

        #ifdef HAVE_IRRLICHT
              if (use_graphics)
              {
                pcs->update_scene();
                pcs->draw_all();
              }
        #endif
    }
    prev_time = rclcpp::Time(odo->header.stamp);
  }

  void place_context_callback(
    const topological_msgs::msg::PlaceContextMatch::SharedPtr match)
  {
    RCLCPP_INFO(this->get_logger(), "PC:place_context_callback id=%d rad=%f matched=%d",
                 match->context_cell_id, match->relative_rad, static_cast<int>(match->matched));

    pc->on_view_template(match->context_cell_id, match->relative_rad);
  
        #ifdef HAVE_IRRLICHT
            if (use_graphics)
            {
              pcs->update_scene();
              pcs->draw_all();
            }
        #endif
  }

  void applyModeChange()
  {
    if (ModeGlobals::getInstance().isMappingMode()) {
      RCLCPP_INFO(get_logger(), "🗺️ PoseCell: MAPPING mode activated");
      // Habilita criação de experiências
    } else if (ModeGlobals::getInstance().isNavigationMode()) {
      RCLCPP_INFO(get_logger(), "🧭 PoseCell: NAVIGATION mode activated");
      // Desabilita criação de experiências
    }
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters)
  {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      
      for (const auto& param : parameters) {
          if (param.get_name() == "pc_vt_inject_energy") {
              double new_value = param.as_double();
              if (pc) {
                  pc->set_vt_inject_energy(new_value);
                  RCLCPP_INFO(this->get_logger(), 
                              "✅ pc_vt_inject_energy updated to: %.6f", new_value);
              }
          }
      }
      
      return result;
  }

  PosecellNetwork* pc = nullptr;
  rclcpp::Publisher<topological_msgs::msg::TopologicalAction>::SharedPtr pub_pc;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry;
  rclcpp::Subscription<topological_msgs::msg::PlaceContextMatch>::SharedPtr sub_place_context;
  rclcpp::Time prev_time{0, 0, RCL_ROS_TIME};
  topological_msgs::msg::TopologicalAction pc_output;
  std::string current_mode_ = "mapping";
  bool holonomic_odometry_ = false;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_subscriber_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_pc_service_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr import_pc_service_;  

#ifdef HAVE_IRRLICHT
  PosecellScene *pcs = nullptr;
  bool use_graphics;
#endif
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PoseCellNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
