#include "utils.h"

#include <rclcpp/rclcpp.hpp>

#include "experience_map.h"
#include <topological_msgs/msg/topological_action.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "experience_map_scene.h"
#include <topological_msgs/msg/topological_map.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>  
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <topological_msgs/msg/keyframe_cloud.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <std_msgs/msg/string.hpp>  
#include "mode_manager/mode_globals.h"  
#include "state_manager.hpp" 


// ============================================
// INCLUDES TO HANDLE MAP EXPORT/IMPORT
// ============================================
#include <std_srvs/srv/empty.hpp>
#include <json/json.h>
#include <fstream>
#include <memory>
#include "map_manager.hpp"
#include <htmslam/srv/import_map.hpp>
#include <htmslam/srv/list_maps.hpp>
#include "htmslam/srv/set_goal.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>

namespace
{
// Translation below this (metres per odometry step) has no reliable direction,
// so the holonomic heading offset is kept at zero.
constexpr double kMinHolonomicTranslation = 1e-4;
}  // namespace


#ifdef HAVE_IRRLICHT
#include "experience_map_scene.h"
#endif

/**
 * @brief ExperienceMapNode - Topological map building and path planning
 * 
 * This ROS2 node maintains a topological map (experience map) of the robot's trajectory
 * as a graph of experiences (nodes) connected by links (edges). It receives actions from
 * the pose cell network to create nodes, create edges, or relocalize. The map is
 * continuously refined using an iterative relaxation algorithm to correct accumulated
 * odometry drift. The node also supports goal-directed navigation by computing shortest
 * paths through the experience graph.
 * 
 * Subscriptions:
 *   - <topic_root>/odom: Receives odometry for dead reckoning between experiences
 *   - <topic_root>/PoseCell/TopologicalAction: Receives actions to update the experience map
 *   - <topic_root>/ExperienceMap/SetGoalPose: Receives goal poses for navigation
 * 
 * Publications:
 *   - <topic_root>/ExperienceMap/Map: Publishes complete topological map (periodic)
 *   - <topic_root>/ExperienceMap/MapMarker: Publishes visualization markers for RViz
 *   - <topic_root>/ExperienceMap/RobotPose: Publishes current robot pose in map frame
 *   - <topic_root>/ExperienceMap/PathToGoal: Publishes planned path to goal
 * 
 * Parameters:
 *   - topic_root: Base topic namespace for all subscriptions and publications
 *   - exp_correction: Correction rate for map relaxation algorithm (default: 0.5)
 *   - exp_loops: Number of relaxation iterations per update (default: 10)
 *   - exp_initial_em_deg: Initial heading in degrees for first experience (default: 90.0)
 *   - enable: Enable Irrlicht graphics visualization (default: true)
 *   - exp_map_size: Size of visualization window (default: 500)
 *   - media_path: Path to media resources for visualization
 *   - image_file: Image file for visualization background
 *   - holonomic_odometry: Use twist.linear.y as well, decoupling the
 *     translation direction from the heading (default: false)
 */
class ExperienceMapNode : public rclcpp::Node
{
  public:
    ExperienceMapNode()
      : Node("experience_map_node")
    {
      // Declare and get parameters
      this->declare_parameter("topic_root", "");
      this->declare_parameter("exp_correction", 0.5);
      this->declare_parameter("exp_loops", 10);
      this->declare_parameter("exp_initial_em_deg", 90.0);
      this->declare_parameter("enable", true);
      this->declare_parameter("exp_map_size", 500);
      this->declare_parameter("media_path", "");
      this->declare_parameter("image_file", "");
      this->declare_parameter("export_filename", "topological_map.json");
      this->declare_parameter("holonomic_odometry", false);
      this->declare_parameter("store_keyframe_clouds", true);
      this->declare_parameter("base_frame", "base_link");
      this->declare_parameter("sensor_mount_xyz", std::vector<double>{0.0, 0.0, 0.3});
      this->declare_parameter("sensor_mount_rpy", std::vector<double>{0.0, 0.0, 0.0});
      this->declare_parameter("global_cloud_voxel_size", 0.30);
      this->declare_parameter("cloud_publish_period", 5.0);
      this->declare_parameter("marker_namespace_chunk", 100);
      this->declare_parameter("marker_refresh_period", 30.0);
      this->declare_parameter("marker_point_size", 0.025);
      this->declare_parameter("keyframe_z_min", 0.2);
      this->declare_parameter("keyframe_z_max", 0.4);

      topic_root_ = this->get_parameter("topic_root").as_string();
      holonomic_odometry_ = this->get_parameter("holonomic_odometry").as_bool();
      store_keyframe_clouds_ = this->get_parameter("store_keyframe_clouds").as_bool();
      base_frame_ = this->get_parameter("base_frame").as_string();
      {
        const auto xyz = this->get_parameter("sensor_mount_xyz").as_double_array();
        const auto rpy = this->get_parameter("sensor_mount_rpy").as_double_array();
        for (std::size_t i = 0; i < 3 && i < xyz.size(); ++i) sensor_mount_xyz_[i] = xyz[i];
        for (std::size_t i = 0; i < 3 && i < rpy.size(); ++i) sensor_mount_rpy_[i] = rpy[i];
      }
      global_cloud_voxel_size_ = this->get_parameter("global_cloud_voxel_size").as_double();
      cloud_publish_period_ = this->get_parameter("cloud_publish_period").as_double();
      marker_namespace_chunk_ = static_cast<int>(
        std::max<int64_t>(1, this->get_parameter("marker_namespace_chunk").as_int()));
      marker_refresh_period_ = this->get_parameter("marker_refresh_period").as_double();
      marker_point_size_ = this->get_parameter("marker_point_size").as_double();
      z_band_min_ = this->get_parameter("keyframe_z_min").as_double();
      z_band_max_ = this->get_parameter("keyframe_z_max").as_double();
      
      // Log all parameters for debugging
      RCLCPP_INFO(this->get_logger(), "ExperienceMap Parameters:");
      RCLCPP_INFO(this->get_logger(), "  topic_root: %s", topic_root_.c_str());
      RCLCPP_INFO(this->get_logger(), "  exp_correction: %f", this->get_parameter("exp_correction").as_double());
      RCLCPP_INFO(this->get_logger(), "  exp_loops: %ld", this->get_parameter("exp_loops").as_int());
      RCLCPP_INFO(this->get_logger(), "  exp_initial_em_deg: %f", this->get_parameter("exp_initial_em_deg").as_double());
      RCLCPP_INFO(this->get_logger(), "  export_filename: %s", this->get_parameter("export_filename").as_string().c_str());
      RCLCPP_INFO(this->get_logger(), "  holonomic_odometry: %s", holonomic_odometry_ ? "true" : "false");

      // Create ExperienceMap with parameters
      em = new ExperienceMap(
        this->get_parameter("exp_correction").as_double(),
        this->get_parameter("exp_loops").as_int(),
        this->get_parameter("exp_initial_em_deg").as_double()
      );

      // Initialize odometry pose to (0,0,0)
      em->resetOdomPose(0.0, 0.0, 0.0);

      // Initialize Map Manager with proper directory
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

      // Initialize Map Manager
      map_manager_ = std::make_shared<htmslam::MapManager>(em, states_dir);
      RCLCPP_INFO(this->get_logger(), "Map Manager initialized");
      RCLCPP_INFO(this->get_logger(), "  Map directory: %s", map_manager_->get_states_dir().c_str());

      // Export/Import services
      export_json_service_ = this->create_service<std_srvs::srv::Empty>(
          "/experience_map/export_json",
          std::bind(&ExperienceMapNode::export_json_callback, this, 
                    std::placeholders::_1, std::placeholders::_2)
      );

      export_yaml_service_ = this->create_service<std_srvs::srv::Empty>(
          "/experience_map/export_yaml",
          std::bind(&ExperienceMapNode::export_yaml_callback, this,
                    std::placeholders::_1, std::placeholders::_2)
      );

      export_binary_service_ = this->create_service<std_srvs::srv::Empty>(
          "/experience_map/export_binary",
          std::bind(&ExperienceMapNode::export_binary_callback, this,
                    std::placeholders::_1, std::placeholders::_2)
      );

      export_all_service_ = this->create_service<std_srvs::srv::Empty>(
          "/experience_map/export_all",
          std::bind(&ExperienceMapNode::export_all_callback, this,
                    std::placeholders::_1, std::placeholders::_2)
      );

      import_map_service_ = this->create_service<htmslam::srv::ImportMap>(
          "/experience_map/import_map",
          std::bind(&ExperienceMapNode::import_map_callback, this,
                    std::placeholders::_1, std::placeholders::_2)
      );

      list_maps_service_ = this->create_service<htmslam::srv::ListMaps>(
          "/experience_map/list_maps",
          std::bind(&ExperienceMapNode::list_maps_callback, this,
                    std::placeholders::_1, std::placeholders::_2)
      );

      RCLCPP_INFO(this->get_logger(), "Map export/import services initialized");

      //NEW: Set goal service
      set_goal_service_ = this->create_service<htmslam::srv::SetGoal>(
      "/experience_map/set_goal",
      std::bind(&ExperienceMapNode::set_goal_callback, this,
                std::placeholders::_1, std::placeholders::_2));
      
      // Initialize publishers
      pub_em = this->create_publisher<topological_msgs::msg::TopologicalMap>(
        topic_root_ + "/ExperienceMap/Map", 10);
      pub_em_markers = this->create_publisher<visualization_msgs::msg::Marker>(
        topic_root_ + "/ExperienceMap/MapMarker", 10);
      pub_pose = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        topic_root_ + "/ExperienceMap/RobotPose", 10);
      pub_goal_path = this->create_publisher<nav_msgs::msg::Path>(
        topic_root_ + "/ExperienceMap/PathToGoal", 10);

      // Publisher para MarkerArray (RViz)
      pub_rviz_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        topic_root_ + "/ExperienceMap/RVizMarkers", 10);

      // Publisher para o marker da experiência atual (NAVIGATION)
      pub_nav_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      topic_root_ + "/ExperienceMap/NavigationMarker", 10);

      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
      RCLCPP_INFO(this->get_logger(), "TF broadcaster initialized");

      if (store_keyframe_clouds_) {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // One marker per experience, so individual clouds can be toggled in RViz.
        auto markers_qos = rclcpp::QoS(10);
        pub_keyframe_markers_ =
          this->create_publisher<visualization_msgs::msg::MarkerArray>(
            topic_root_ + "/ExperienceMap/KeyframeMarkers", markers_qos);

        // Overview cloud: published on change, kept for late subscribers.
        auto cloud_qos = rclcpp::QoS(1);
        cloud_qos.transient_local();
        pub_keyframe_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
          topic_root_ + "/ExperienceMap/KeyframeClouds", cloud_qos);

        sub_keyframe_cloud_ = this->create_subscription<topological_msgs::msg::KeyframeCloud>(
          topic_root_ + "/place_context_cloud", 10,
          std::bind(&ExperienceMapNode::keyframe_cloud_callback, this,
                    std::placeholders::_1));

        cloud_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(200),
          std::bind(&ExperienceMapNode::cloud_timer_callback, this));

        RCLCPP_INFO(
          this->get_logger(),
          "Keyframe clouds: storing in '%s' frame, publishing '%s' (transient_local) "
          "and '%s'",
          base_frame_.c_str(),
          (topic_root_ + "/ExperienceMap/KeyframeClouds").c_str(),
          (topic_root_ + "/ExperienceMap/KeyframeMarkers").c_str());
        if (z_band_max_ > z_band_min_) {
          RCLCPP_INFO(
            this->get_logger(),
            "Keyframe clouds: keeping z in [%.2f, %.2f] m of '%s' "
            "(floor/ceiling are discarded)",
            z_band_min_, z_band_max_, base_frame_.c_str());
        } else {
          RCLCPP_INFO(this->get_logger(),
                      "Keyframe clouds: z band disabled (keyframe_z_max <= keyframe_z_min)");
        }
      }
      
      // Initialize subscribers
      // Sensor data QoS: odometry sources (T265, bags) usually publish best_effort,
      // which a reliable subscriber cannot receive.
      sub_odometry = this->create_subscription<nav_msgs::msg::Odometry>(
        topic_root_ + "/odom", rclcpp::SensorDataQoS(),
        std::bind(&ExperienceMapNode::odo_callback, this, std::placeholders::_1));
      
      sub_action = this->create_subscription<topological_msgs::msg::TopologicalAction>(
        topic_root_ + "/PoseCell/TopologicalAction", 10,
        std::bind(&ExperienceMapNode::action_callback, this, std::placeholders::_1));
      
      sub_goal = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        topic_root_ + "/ExperienceMap/SetGoalPose", 10,
        std::bind(&ExperienceMapNode::set_goal_pose_callback, this, std::placeholders::_1));

    // ADICIONA O SUBSCRIBER DO MODO
      mode_subscriber_ = this->create_subscription<std_msgs::msg::String>(
        "/system_mode", 10,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          current_mode_ = msg->data;
          ModeGlobals::getInstance().setMode(current_mode_);
          RCLCPP_INFO(this->get_logger(), "🔀 ExperienceMap mode changed to: %s", 
                      current_mode_.c_str());
          applyModeChange();
        });
      
      // Inicializa o modo
      std::string initial_mode = ModeGlobals::getInstance().getMode();
      RCLCPP_INFO(this->get_logger(), "📝 ExperienceMap initial mode: %s", 
                  initial_mode.c_str());
      
  #ifdef HAVE_IRRLICHT
      use_graphics = this->get_parameter("enable").as_bool();
      if (use_graphics)
      {
        ems = new ExperienceMapScene(
          this->get_parameter("exp_map_size").as_int(),
          this->get_parameter("media_path").as_string(),
          this->get_parameter("image_file").as_string(),
          em
        );
      }
  #endif
    }
    
    ~ExperienceMapNode()
    {
      if (em != nullptr)
        delete em;
  #ifdef HAVE_IRRLICHT
      if (ems != nullptr)
        delete ems; 
  #endif
    }

    // ============================================
    // CALLBACKS OF THE SERVICES FOR EXPORTING AND IMPORTING MAPS
    // ============================================

    void export_json_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (map_manager_) {
            htmslam::MapMetadata metadata;
            metadata.description = "NeoSLAM Topological Map (JSON)";
            std::string timestamp = std::to_string(this->now().seconds());
            metadata.custom_params["topic_root"] = topic_root_;
            map_manager_->export_json(topic_root_ + "_map.json", metadata);
        }
    }

    void export_yaml_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (map_manager_) {
            htmslam::MapMetadata metadata;
            metadata.description = "NeoSLAM Topological Map (YAML)";
            std::string timestamp = std::to_string(this->now().seconds());
            map_manager_->export_yaml("map_" + timestamp + ".yaml", metadata);
        }
    }

    void export_binary_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (map_manager_) {
            htmslam::MapMetadata metadata;
            metadata.description = "NeoSLAM Topological Map (Binary)";
            std::string timestamp = std::to_string(this->now().seconds());
            map_manager_->export_binary("map_" + timestamp + ".bin", metadata);
        }
    }

    void export_all_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (map_manager_) {
            htmslam::MapMetadata metadata;
            metadata.description = "NeoSLAM Topological Map (All Formats)";
            std::string timestamp = std::to_string(this->now().seconds());
            map_manager_->export_all("map_" + timestamp, metadata);
        }
    }

    void import_map_callback(
        const std::shared_ptr<htmslam::srv::ImportMap::Request> request,
        std::shared_ptr<htmslam::srv::ImportMap::Response> response)
    {
        if (map_manager_) {
          bool success = map_manager_->import_map(request->filename);
          response->success = success;
          response->message = success ? "Map imported successfully" : "Failed to import map";

          if (success) {
            // ============================================
            // Publish the map after import
            // ============================================
            RCLCPP_INFO(this->get_logger(), "📡 Publishing map after import...");
            publishMap(true);  // Publish the complete map

            // If in NAVIGATION mode, adjust the odometry pose
            if (ModeGlobals::getInstance().isNavigationMode()) {
                auto [x, y, th] = em->getCurrentPose();
                em->resetOdomPose(x, y, th);
                RCLCPP_INFO(this->get_logger(), "📌 Odom pose set to: x=%.3f y=%.3f th=%.3f", 
                            x, y, th);
            }
          }
        } else {
          response->success = false;
          response->message = "Map manager not initialized";
        }
    }

    void list_maps_callback(
        const std::shared_ptr<htmslam::srv::ListMaps::Request> request,
        std::shared_ptr<htmslam::srv::ListMaps::Response> response)
    {
        (void)request;
        
        if (map_manager_) {
            auto maps = map_manager_->list_maps();
            response->maps = maps;
            response->count = maps.size();
            response->directory = map_manager_->get_states_dir();
        } else {
            response->count = 0;
            response->directory = "";
        }
    }
  
    // Service callback to set a new goal
    void set_goal_callback(
      const std::shared_ptr<htmslam::srv::SetGoal::Request> request,
      std::shared_ptr<htmslam::srv::SetGoal::Response> response)
    {
      if (em->get_num_experiences() == 0) {
        response->success = false;
        response->message = "Map is empty. Run mapping mode first!";
        RCLCPP_WARN(this->get_logger(), "❌ Cannot set goal: Map is empty");
        return;
      }
      
      // Add the goal to the experience map
      em->add_goal(request->x, request->y);
      response->success = true;
      response->message = "Goal set at (" + std::to_string(request->x) + ", " + 
                          std::to_string(request->y) + ")";
      response->goal_id = em->get_current_goal_id();
      
      RCLCPP_INFO(this->get_logger(), "🎯 Goal set at (%.2f, %.2f)", 
                  request->x, request->y);
      
      // Replan the path to the new goal
      em->calculate_path_to_goal(rclcpp::Clock().now().seconds());
      em->get_goal_waypoint();
    }

    void publishMap(bool force = false) {
        // Publish the map in TopologicalMap message format
        topological_msgs::msg::TopologicalMap em_map;
        em_map.header.stamp = this->now();
        em_map.header.frame_id = "map";
        em_map.node_count = em->get_num_experiences();
        em_map.node.resize(em->get_num_experiences());
        
        for (int i = 0; i < em->get_num_experiences(); i++) {
            em_map.node[i].id = em->get_experience(i)->id;
            em_map.node[i].pose.pose.position.x = em->get_experience(i)->x_m;
            em_map.node[i].pose.pose.position.y = em->get_experience(i)->y_m;
            em_map.node[i].pose.header.stamp.sec = em->get_experience(i)->seconds;
            em_map.node[i].pose.header.stamp.nanosec = em->get_experience(i)->nanoseconds;
            em_map.node[i].pose.header.frame_id = "map";
            tf2::Quaternion q;
            q.setRPY(0, 0, em->get_experience(i)->th_rad);
            em_map.node[i].pose.pose.orientation = tf2::toMsg(q);
        }

        em_map.edge_count = em->get_num_links();
        em_map.edge.resize(em->get_num_links());
        
        for (int i = 0; i < em->get_num_links(); i++) {
            em_map.edge[i].id = i;
            em_map.edge[i].source_id = em->get_link(i)->exp_from_id;
            em_map.edge[i].destination_id = em->get_link(i)->exp_to_id;
            em_map.edge[i].duration = rclcpp::Duration::from_seconds(em->get_link(i)->delta_time_s);
            em_map.edge[i].transform.translation.x = em->get_link(i)->d * cos(em->get_link(i)->heading_rad);
            em_map.edge[i].transform.translation.y = em->get_link(i)->d * sin(em->get_link(i)->heading_rad);
            
            tf2::Quaternion q;
            q.setRPY(0, 0, em->get_link(i)->facing_rad);
            em_map.edge[i].transform.rotation = tf2::toMsg(q);
        }
        
        pub_em->publish(em_map);
        
        // Publish RViz markers
        publishRVizMarkers();
        
        RCLCPP_INFO(this->get_logger(), "📡 Map published: %d nodes, %d edges", 
                    em->get_num_experiences(), em->get_num_links());
    }

    void publishRVizMarkers() {
        visualization_msgs::msg::MarkerArray rviz_marker_array;
        
        visualization_msgs::msg::Marker nodes_marker;
        nodes_marker.header.frame_id = "map";
        nodes_marker.header.stamp = this->now();
        nodes_marker.ns = "nodes";
        nodes_marker.id = 0;
        nodes_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        nodes_marker.action = visualization_msgs::msg::Marker::ADD;
        nodes_marker.scale.x = 0.05;
        nodes_marker.scale.y = 0.05;
        nodes_marker.scale.z = 0.05;
        nodes_marker.color.r = 0.0;
        nodes_marker.color.g = 1.0;
        nodes_marker.color.b = 0.0;
        nodes_marker.color.a = 1.0;
        
        for (int i = 0; i < em->get_num_experiences(); i++) {
            Experience* exp = em->get_experience(i);
            if (exp == nullptr) continue;
            
            geometry_msgs::msg::Point p;
            p.x = exp->x_m;
            p.y = exp->y_m;
            p.z = 0.0;
            nodes_marker.points.push_back(p);
        }
        rviz_marker_array.markers.push_back(nodes_marker);
        
        visualization_msgs::msg::Marker edges_marker;
        edges_marker.header.frame_id = "map";
        edges_marker.header.stamp = this->now();
        edges_marker.ns = "edges";
        edges_marker.id = 1;
        edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        edges_marker.action = visualization_msgs::msg::Marker::ADD;
        edges_marker.scale.x = 0.01;
        edges_marker.color.r = 1.0;
        edges_marker.color.g = 1.0;
        edges_marker.color.b = 1.0;
        edges_marker.color.a = 1.0;
        
        for (int i = 0; i < em->get_num_links(); i++) {
            Link* link = em->get_link(i);
            if (link == nullptr) continue;
            
            Experience* from = em->get_experience(link->exp_from_id);
            Experience* to = em->get_experience(link->exp_to_id);
            if (from == nullptr || to == nullptr) continue;
            
            geometry_msgs::msg::Point p1, p2;
            p1.x = from->x_m;
            p1.y = from->y_m;
            p1.z = 0.0;
            p2.x = to->x_m;
            p2.y = to->y_m;
            p2.z = 0.0;
            
            edges_marker.points.push_back(p1);
            edges_marker.points.push_back(p2);
        }
        rviz_marker_array.markers.push_back(edges_marker);
        
        // Current node in red
        visualization_msgs::msg::Marker current_marker;
        current_marker.header.frame_id = "map";
        current_marker.header.stamp = this->now();
        current_marker.ns = "current";
        current_marker.id = 2;
        current_marker.type = visualization_msgs::msg::Marker::SPHERE;
        current_marker.action = visualization_msgs::msg::Marker::ADD;
        current_marker.scale.x = 0.15;
        current_marker.scale.y = 0.15;
        current_marker.scale.z = 0.15;
        current_marker.color.r = 1.0;
        current_marker.color.g = 0.0;
        current_marker.color.b = 0.0;
        current_marker.color.a = 1.0;
        
        Experience* current = em->get_experience(em->get_current_id());
        if (current != nullptr) {
            current_marker.pose.position.x = current->x_m;
            current_marker.pose.position.y = current->y_m;
            current_marker.pose.position.z = 0.0;
            rviz_marker_array.markers.push_back(current_marker);
        }
        
        pub_rviz_markers_->publish(rviz_marker_array);
    }

  private:
    void odo_callback(const nav_msgs::msg::Odometry::SharedPtr odo)
    {
      RCLCPP_DEBUG(this->get_logger(), "EM:odo_callback v=%f r=%f",
                  odo->twist.twist.linear.x, odo->twist.twist.angular.z);

      if (prev_time.seconds() > 0)
      {
        double time_diff = (rclcpp::Time(odo->header.stamp) - prev_time).seconds();
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
          em->on_odo_holonomic(speed, dheading, odo->twist.twist.angular.z, time_diff);
        }
        else
        {
          em->on_odo(odo->twist.twist.linear.x, odo->twist.twist.angular.z, time_diff);
        }
        
        // If in NAVIGATION mode, publish the odometry pose (integrated)
        if (ModeGlobals::getInstance().isNavigationMode()) {
          auto [x, y, th] = em->getOdomPose();
          publishRobotPoseFromOdometry(x, y, th, odo->header.stamp);
        }
        
        if (em->get_current_goal_id() >= 0)
        { 
          // RCLCPP_DEBUG(this->get_logger(), "📊 [odo] Recalculating path...");
          // RCLCPP_DEBUG(this->get_logger(), "📊 Current exp: %d, Goal: %d", 
          //             em->get_current_id(), em->get_current_goal_id());

          // Force the time_from_current_s to DBL_MAX for all experiences to force Dijkstra recalculation
          for (int id = 0; id < em->get_num_experiences(); id++) {
              Experience* exp = em->get_experience(id);
              if (exp != nullptr) {
                  exp->time_from_current_s = DBL_MAX;
                  exp->goal_to_current = -1;
                  exp->current_to_goal = -1;
              }
          }

          em->resetGoalTimeout();  
          
          double dx = em->get_experience(em->get_current_goal_id())->x_m - em->get_experience(em->get_current_id())->x_m;
          double dy = em->get_experience(em->get_current_goal_id())->y_m - em->get_experience(em->get_current_id())->y_m;
          double real_dist = sqrt(dx*dx + dy*dy);
          
          RCLCPP_INFO(this->get_logger(), "📏 Real distance to goal: %.3f m", real_dist);

          prev_goal_update = rclcpp::Time(odo->header.stamp);
          em->calculate_path_to_goal(rclcpp::Time(odo->header.stamp).seconds());

          nav_msgs::msg::Path path;
          if (em->get_current_goal_id() >= 0)
          {
            em->get_goal_waypoint();

            geometry_msgs::msg::PoseStamped pose;
            path.header.stamp = this->now();
            path.header.frame_id = "map";

            path.poses.clear();
            unsigned int trace_exp_id = em->get_goals()[0];
            while (trace_exp_id != em->get_goal_path_final_exp())
            {
              pose.header.stamp = this->now();
              pose.header.frame_id = "map";
              pose.pose.position.x = em->get_experience(trace_exp_id)->x_m;
              pose.pose.position.y = em->get_experience(trace_exp_id)->y_m;
              path.poses.push_back(pose);

              trace_exp_id = em->get_experience(trace_exp_id)->goal_to_current;
            }

            pub_goal_path->publish(path);
          }
          else
          {
            path.header.stamp = this->now();
            path.header.frame_id = "map";
            path.poses.clear();
            pub_goal_path->publish(path);
          }
        }
      }

      prev_time = rclcpp::Time(odo->header.stamp);

      std::cout << "Current Exp: " << em->get_current_id() 
                << " | Total Exps: " << em->get_num_experiences() 
                << " | Total actions: " << action_counter 
                << std::endl;
      std::cout.flush();
    }

    void action_callback(const topological_msgs::msg::TopologicalAction::SharedPtr action)
    { 
      
      /// Em NAVIGATION, processa apenas ações de loop closure (CREATE_EDGE)
      if (ModeGlobals::getInstance().isNavigationMode()) {
          // Verifica se é uma ação de loop closure
          if (action->action == topological_msgs::msg::TopologicalAction::CREATE_EDGE) {
              RCLCPP_INFO(this->get_logger(), 
                          "🧭 NAVIGATION: Loop closure detected! Correcting pose to experience %d", 
                          action->dest_id);
              
              // Correct the robot's pose to the experience's pose
              bool success = em->correctPoseToExperience(action->dest_id, action->relative_rad);
              
              if (success) {
                auto [x, y, th] = em->getOdomPose();
                publishRobotPoseFromOdometry(x, y, th, action->header.stamp);
                
                RCLCPP_INFO(this->get_logger(), 
                            "✅ Pose corrected to: x=%.3f y=%.3f th=%.3f", 
                            x, y, th);

                
                visualization_msgs::msg::MarkerArray nav_marker_array;
                
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp = action->header.stamp;
                marker.ns = "navigation_marker";
                marker.id = 999;
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.scale.x = 0.20;
                marker.scale.y = 0.20;
                marker.scale.z = 0.20;
                marker.color.r = 1.0;
                marker.color.g = 0.5;
                marker.color.b = 0.0;
                marker.color.a = 1.0;
                marker.pose.position.x = x;
                marker.pose.position.y = y;
                marker.pose.position.z = 0.0;
                marker.pose.orientation.w = 1.0;
                
                nav_marker_array.markers.push_back(marker);
                
                pub_rviz_markers_->publish(nav_marker_array);
                
                RCLCPP_INFO(this->get_logger(), "🔴 NAVIGATION MARKER published at (%.2f, %.2f)", x, y);

                // ============================================
                // NEW: Replan the path to the goal after loop closure
                // ============================================
                if (em->get_current_goal_id() >= 0) {
                  RCLCPP_INFO(this->get_logger(), "🔄 Replanning path after loop closure");
                  // RCLCPP_INFO(this->get_logger(), "📊 Current exp: %d, Goal exp: %d", 
                  //             em->get_current_id(), em->get_current_goal_id());
                  // RCLCPP_INFO(this->get_logger(), "📊 Total experiences: %d", em->get_num_experiences());
                  
                  // Reset distances
                  for (int id = 0; id < em->get_num_experiences(); id++) {
                      Experience* exp = em->get_experience(id);
                      if (exp != nullptr) {
                          exp->time_from_current_s = DBL_MAX;
                          exp->goal_to_current = -1;
                          exp->current_to_goal = -1;
                      }
                  }
                  em->calculate_path_to_goal(rclcpp::Time(action->header.stamp).seconds());
                  em->get_goal_waypoint();
                  
                  nav_msgs::msg::Path path;
                  path.header.stamp = this->now();
                  path.header.frame_id = "map";
                  path.poses.clear();
                  
                  // Rebuild the path to the goal
                  if (em->get_current_goal_id() >= 0) {
                      unsigned int trace_exp_id = em->get_goals()[0];
                      while (trace_exp_id != em->get_goal_path_final_exp()) {
                          geometry_msgs::msg::PoseStamped pose;
                          pose.header.stamp = this->now();
                          pose.header.frame_id = "map";
                          pose.pose.position.x = em->get_experience(trace_exp_id)->x_m;
                          pose.pose.position.y = em->get_experience(trace_exp_id)->y_m;
                          path.poses.push_back(pose);
                          
                          trace_exp_id = em->get_experience(trace_exp_id)->goal_to_current;
                      }
                  }
                  
                  pub_goal_path->publish(path);
                  RCLCPP_INFO(this->get_logger(), "📡 Path published with %zu poses", path.poses.size());
              }

            } else {
                RCLCPP_WARN(this->get_logger(), 
                            "❌ Failed to correct pose to experience %d", 
                            action->dest_id);
            }
          } else {
              
              RCLCPP_DEBUG(this->get_logger(), 
                          "EM: NAVIGATION mode - ignoring action %d", action->action);
          }
          return;
      }

      action_counter++;
          
      RCLCPP_INFO(this->get_logger(), "EM:action_callback action=%d src=%d dst=%d vt_id=%d",
                  action->action, action->src_id, action->dest_id, action->vt_id);

      switch (action->action)
      {
        case topological_msgs::msg::TopologicalAction::CREATE_NODE:
          em->on_create_experience(action->dest_id, action->header.stamp.sec, action->header.stamp.nanosec, action->vt_id);
          em->on_set_experience(action->dest_id, 0);
          on_experience_created(action->dest_id, action->vt_id);
          break;

        case topological_msgs::msg::TopologicalAction::CREATE_EDGE:
          em->on_create_link(action->src_id, action->dest_id, action->relative_rad);
          em->on_set_experience(action->dest_id, action->relative_rad);
          break;

        case topological_msgs::msg::TopologicalAction::SET_NODE:
          em->on_set_experience(action->dest_id, action->relative_rad);
          break;
      }

      em->iterate();

      geometry_msgs::msg::PoseStamped pose_output;
      pose_output.header.stamp = action->header.stamp;
      pose_output.header.frame_id = "map";
      pose_output.pose.position.x = em->get_experience(em->get_current_id())->x_m;
      pose_output.pose.position.y = em->get_experience(em->get_current_id())->y_m;
      pose_output.pose.position.z = 0;
      
      tf2::Quaternion q;
      q.setRPY(0, 0, em->get_experience(em->get_current_id())->th_rad);
      pose_output.pose.orientation = tf2::toMsg(q);
      
      pub_pose->publish(pose_output);

      // ============================================
      // Publish the robot's pose as a TF transform for visualization in RViz
      // ============================================
      geometry_msgs::msg::TransformStamped tf_transform;
      tf_transform.header.stamp = action->header.stamp;
      tf_transform.header.frame_id = "map";
      tf_transform.child_frame_id = "base_link";

      Experience* current_exp = em->get_experience(em->get_current_id());
      if (current_exp != nullptr) {
          tf_transform.transform.translation.x = current_exp->x_m;
          tf_transform.transform.translation.y = current_exp->y_m;
          tf_transform.transform.translation.z = 0.0;
          
          tf2::Quaternion q_tf;
          q_tf.setRPY(0, 0, current_exp->th_rad);
          tf_transform.transform.rotation = tf2::toMsg(q_tf);
          
          tf_broadcaster_->sendTransform(tf_transform);
      }

      if ((rclcpp::Time(action->header.stamp) - prev_pub_time).seconds() > 30.0)
      {
        prev_pub_time = rclcpp::Time(action->header.stamp);

        topological_msgs::msg::TopologicalMap em_map;
        em_map.header.stamp = action->header.stamp;
        em_map.header.frame_id = "map";
        em_map.node_count = em->get_num_experiences();
        em_map.node.resize(em->get_num_experiences());
        
        for (int i = 0; i < em->get_num_experiences(); i++)
        {
          em_map.node[i].id = em->get_experience(i)->id;
          em_map.node[i].pose.pose.position.x = em->get_experience(i)->x_m;
          em_map.node[i].pose.pose.position.y = em->get_experience(i)->y_m;
          em_map.node[i].pose.header.stamp.sec = em->get_experience(i)->seconds;
          em_map.node[i].pose.header.stamp.nanosec = em->get_experience(i)->nanoseconds;
          em_map.node[i].pose.header.frame_id = "map";
          tf2::Quaternion q;
          q.setRPY(0, 0, em->get_experience(i)->th_rad);
          em_map.node[i].pose.pose.orientation = tf2::toMsg(q);
        }

        em_map.edge_count = em->get_num_links();
        em_map.edge.resize(em->get_num_links());
        
        for (int i = 0; i < em->get_num_links(); i++)
        {
          em_map.edge[i].id = i;
          em_map.edge[i].source_id = em->get_link(i)->exp_from_id;
          em_map.edge[i].destination_id = em->get_link(i)->exp_to_id;
          em_map.edge[i].duration = rclcpp::Duration::from_seconds(em->get_link(i)->delta_time_s);
          em_map.edge[i].transform.translation.x = em->get_link(i)->d * cos(em->get_link(i)->heading_rad);
          em_map.edge[i].transform.translation.y = em->get_link(i)->d * sin(em->get_link(i)->heading_rad);
          
          tf2::Quaternion q;
          q.setRPY(0, 0, em->get_link(i)->facing_rad);
          em_map.edge[i].transform.rotation = tf2::toMsg(q);
        }
        
        pub_em->publish(em_map);
      }

      visualization_msgs::msg::Marker em_marker;
      em_marker.header.stamp = this->now();
      em_marker.header.frame_id = "map";
      em_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
      em_marker.points.resize(em->get_num_links() * 2);
      em_marker.action = visualization_msgs::msg::Marker::ADD;
      em_marker.scale.x = 0.01;
      em_marker.color.a = 1;
      em_marker.ns = "em";
      em_marker.id = 0;
      em_marker.pose.orientation.w = 1;
      
      for (int i = 0; i < em->get_num_links(); i++)
      {
        em_marker.points[i * 2].x = em->get_experience(em->get_link(i)->exp_from_id)->x_m;
        em_marker.points[i * 2].y = em->get_experience(em->get_link(i)->exp_from_id)->y_m;
        em_marker.points[i * 2].z = 0;
        em_marker.points[i * 2 + 1].x = em->get_experience(em->get_link(i)->exp_to_id)->x_m;
        em_marker.points[i * 2 + 1].y = em->get_experience(em->get_link(i)->exp_to_id)->y_m;
        em_marker.points[i * 2 + 1].z = 0;
      }

      pub_em_markers->publish(em_marker);

      // Publicar markers to RViz
      visualization_msgs::msg::MarkerArray rviz_marker_array;
      
      // 1. Nós como esferas verdes
      visualization_msgs::msg::Marker nodes_marker;
      nodes_marker.header.frame_id = "map";
      nodes_marker.header.stamp = this->now();
      nodes_marker.ns = "nodes";
      nodes_marker.id = 0;
      nodes_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      nodes_marker.action = visualization_msgs::msg::Marker::ADD;
      nodes_marker.scale.x = 0.05;
      nodes_marker.scale.y = 0.05;
      nodes_marker.scale.z = 0.05;
      nodes_marker.color.r = 0.0;
      nodes_marker.color.g = 1.0;
      nodes_marker.color.b = 0.0;
      nodes_marker.color.a = 1.0;
      
      for (int i = 0; i < em->get_num_experiences(); i++) {
          Experience* exp = em->get_experience(i);
          if (exp == nullptr) continue;
          
          geometry_msgs::msg::Point p;
          p.x = exp->x_m;
          p.y = exp->y_m;
          p.z = 0.0;
          nodes_marker.points.push_back(p);
      }
      rviz_marker_array.markers.push_back(nodes_marker);
      
      visualization_msgs::msg::Marker edges_marker;
      edges_marker.header.frame_id = "map";
      edges_marker.header.stamp = this->now();
      edges_marker.ns = "edges";
      edges_marker.id = 1;
      edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
      edges_marker.action = visualization_msgs::msg::Marker::ADD;
      edges_marker.scale.x = 0.01;
      edges_marker.color.r = 1.0;
      edges_marker.color.g = 1.0;
      edges_marker.color.b = 1.0;
      edges_marker.color.a = 1.0;
      
      for (int i = 0; i < em->get_num_links(); i++) {
          Link* link = em->get_link(i);
          if (link == nullptr) continue;
          
          Experience* from = em->get_experience(link->exp_from_id);
          Experience* to = em->get_experience(link->exp_to_id);
          if (from == nullptr || to == nullptr) continue;
          
          geometry_msgs::msg::Point p1, p2;
          p1.x = from->x_m;
          p1.y = from->y_m;
          p1.z = 0.0;
          p2.x = to->x_m;
          p2.y = to->y_m;
          p2.z = 0.0;
          
          edges_marker.points.push_back(p1);
          edges_marker.points.push_back(p2);
      }
      rviz_marker_array.markers.push_back(edges_marker);
      
      visualization_msgs::msg::Marker current_marker;
      current_marker.header.frame_id = "map";
      current_marker.header.stamp = this->now();
      current_marker.ns = "current";
      current_marker.id = 2;
      current_marker.type = visualization_msgs::msg::Marker::SPHERE;
      current_marker.action = visualization_msgs::msg::Marker::ADD;
      current_marker.scale.x = 0.20;
      current_marker.scale.y = 0.20;
      current_marker.scale.z = 0.20;
      current_marker.color.r = 1.0;
      current_marker.color.g = 0.0;
      current_marker.color.b = 0.0;
      current_marker.color.a = 1.0;
      
      Experience* current = em->get_experience(em->get_current_id());
      if (current != nullptr) {
          current_marker.pose.position.x = current->x_m;
          current_marker.pose.position.y = current->y_m;
          current_marker.pose.position.z = 0.0;
          rviz_marker_array.markers.push_back(current_marker);
      }
      
      // Publish the MarkerArray
      pub_rviz_markers_->publish(rviz_marker_array);

  #ifdef HAVE_IRRLICHT
      if (use_graphics)
      {
        ems->update_scene();
        ems->draw_all();
      }
  #endif
    }

    void set_goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr pose)
    {
      if (pose->header.frame_id == "map") {
        em->add_goal(pose->pose.position.x, pose->pose.position.y);
      } else {
        RCLCPP_WARN(this->get_logger(), "Goal pose frame_id is '%s', expected 'map'", 
                    pose->header.frame_id.c_str());
        em->add_goal(pose->pose.position.x, pose->pose.position.y);
      }
    }

    void publishRobotPoseFromOdometry(double x, double y, double th, 
                                    const builtin_interfaces::msg::Time& stamp)
    {
      // Publish the robot's pose in the "map" frame
      geometry_msgs::msg::PoseStamped pose_output;
      pose_output.header.stamp = stamp;
      pose_output.header.frame_id = "map";
      pose_output.pose.position.x = x;
      pose_output.pose.position.y = y;
      pose_output.pose.position.z = 0.0;
      
      tf2::Quaternion q;
      q.setRPY(0, 0, th);
      pose_output.pose.orientation = tf2::toMsg(q);
      
      pub_pose->publish(pose_output);

      // Publish the TF transformation
      geometry_msgs::msg::TransformStamped tf_transform;
      tf_transform.header.stamp = stamp;
      tf_transform.header.frame_id = "map";
      tf_transform.child_frame_id = "base_link";
      tf_transform.transform.translation.x = x;
      tf_transform.transform.translation.y = y;
      tf_transform.transform.translation.z = 0.0;
      tf_transform.transform.rotation = tf2::toMsg(q);
      
      tf_broadcaster_->sendTransform(tf_transform);
    }
    
    void applyModeChange()
    {
      if (ModeGlobals::getInstance().isMappingMode()) {
        RCLCPP_INFO(this->get_logger(), "🗺️ ExperienceMap: MAPPING mode");
        em->useExperiencePose();
        
        // Mantém os goals ativos
        if (em->get_current_goal_id() >= 0) {
          RCLCPP_INFO(this->get_logger(), "🎯 Keeping current goal active");
        }
        
      } else if (ModeGlobals::getInstance().isNavigationMode()) {
        RCLCPP_INFO(this->get_logger(), "🧭 ExperienceMap: NAVIGATION mode");
        
        auto [x, y, th] = em->getCurrentPose();
        em->resetOdomPose(x, y, th);
        em->useExperiencePose();
        
        RCLCPP_INFO(this->get_logger(), "📌 Odom pose initialized at: x=%.3f y=%.3f th=%.3f", 
                    x, y, th);
        
        
        if (em->get_current_goal_id() >= 0) {
          RCLCPP_INFO(this->get_logger(), "🔄 Replanning path after mode change");
          em->calculate_path_to_goal(rclcpp::Clock().now().seconds());
          em->get_goal_waypoint();
        }
      }
    }

    // ============================================================
    // Private members
    // ============================================================
    
    // Experience Map 
    ExperienceMap* em = nullptr;  
    
    // ============================================
    // KEYFRAME CLOUDS
    // Each experience keeps the cloud that created its visual template, stored
    // in the robot base frame at the moment of creation. The map pose is applied
    // only when publishing, so the clouds follow loop-closure corrections.
    // ============================================

    /// Sensor -> base transform: TF first, static parameters as fallback.
    tf2::Transform lookup_sensor_mount(const std_msgs::msg::Header & header)
    {
      try {
        const auto tf_msg = tf_buffer_->lookupTransform(
          base_frame_, header.frame_id, header.stamp, rclcpp::Duration::from_seconds(0.05));
        tf2::Transform mount;
        tf2::fromMsg(tf_msg.transform, mount);
        if (!mount_logged_) {
          mount_logged_ = true;
          RCLCPP_INFO(this->get_logger(),
                      "Keyframe clouds: using TF '%s' <- '%s' (z=%.3f)",
                      base_frame_.c_str(), header.frame_id.c_str(),
                      mount.getOrigin().z());
        }
        return mount;
      } catch (const tf2::TransformException &) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this, 10000,
          "Keyframe clouds: TF '%s' <- '%s' unavailable, using sensor_mount_xyz/rpy "
          "parameters",
          base_frame_.c_str(), header.frame_id.c_str());
      }
      tf2::Quaternion q;
      q.setRPY(sensor_mount_rpy_[0], sensor_mount_rpy_[1], sensor_mount_rpy_[2]);
      return tf2::Transform(
        q, tf2::Vector3(sensor_mount_xyz_[0], sensor_mount_xyz_[1], sensor_mount_xyz_[2]));
    }

    void keyframe_cloud_callback(const topological_msgs::msg::KeyframeCloud::SharedPtr msg)
    {
      const auto & cloud = msg->cloud;
      const std::size_t n = static_cast<std::size_t>(cloud.width) * cloud.height;
      if (n == 0 || cloud.point_step < 12 || cloud.data.size() < n * cloud.point_step) {
        return;
      }

      const tf2::Transform mount = lookup_sensor_mount(cloud.header);
      KeyframeCloud data;
      data.xyz.reserve(n * 3);
      const float * src = reinterpret_cast<const float *>(cloud.data.data());
      const std::size_t stride = cloud.point_step / sizeof(float);
      const bool z_band_enabled = z_band_max_ > z_band_min_;
      for (std::size_t i = 0; i < n; ++i) {
        const tf2::Vector3 p(src[i * stride + 0], src[i * stride + 1], src[i * stride + 2]);
        const tf2::Vector3 q = mount * p;
        // Keep only the horizontal slice of interest: floor and ceiling points
        // dominate the cloud and add nothing to the place visualisation.
        if (z_band_enabled && (q.z() < z_band_min_ || q.z() > z_band_max_)) {
          continue;
        }
        data.xyz.push_back(static_cast<float>(q.x()));
        data.xyz.push_back(static_cast<float>(q.y()));
        data.xyz.push_back(static_cast<float>(q.z()));
      }

      if (data.xyz.empty()) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this, 5000,
          "Keyframe cloud for vt=%u is empty after the z band [%.2f, %.2f] m in '%s'; "
          "check keyframe_z_min/keyframe_z_max",
          msg->vt_id, z_band_min_, z_band_max_, base_frame_.c_str());
      } else {
        RCLCPP_INFO_THROTTLE(
          this->get_logger(), *this, 5000,
          "Keyframe cloud vt=%u: %zu -> %zu points in z band [%.2f, %.2f] m",
          msg->vt_id, n, data.size(), z_band_min_, z_band_max_);
      }

      keyframe_clouds_[msg->vt_id] = std::move(data);
      clouds_dirty_ = true;

      // Experiences created before their cloud arrived.
      for (auto it = pending_experiences_.begin(); it != pending_experiences_.end(); ) {
        if (it->second == msg->vt_id) {
          publish_experience_marker(it->first);
          it = pending_experiences_.erase(it);
        } else {
          ++it;
        }
      }
    }

    void on_experience_created(unsigned int exp_id, unsigned int vt_id)
    {
      if (!store_keyframe_clouds_) {
        return;
      }
      if (keyframe_clouds_.count(vt_id) > 0) {
        publish_experience_marker(exp_id);
      } else {
        pending_experiences_[exp_id] = vt_id;
      }
    }

    static void hsv_to_rgb(float h, float s, float v, float & r, float & g, float & b)
    {
      const float i = std::floor(h * 6.0f);
      const float f = h * 6.0f - i;
      const float p = v * (1.0f - s);
      const float q = v * (1.0f - f * s);
      const float t = v * (1.0f - (1.0f - f) * s);
      switch (static_cast<int>(i) % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
      }
    }

    /// One POINTS marker per experience. The marker pose carries the map pose,
    /// so RViz applies map corrections to the stored local points.
    void publish_experience_marker(unsigned int exp_id)
    {
      Experience * exp = em->get_experience(exp_id);
      if (exp == nullptr) {
        return;
      }
      const auto it = keyframe_clouds_.find(static_cast<uint32_t>(exp->vt_id));
      if (it == keyframe_clouds_.end()) {
        return;
      }
      const KeyframeCloud & cloud = it->second;
      const std::size_t n = cloud.size();

      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = this->now();
      const int chunk = marker_namespace_chunk_;
      const int start = (static_cast<int>(exp_id) / chunk) * chunk;
      marker.ns = "exp_cloud_" + std::to_string(start) + "_" +
        std::to_string(start + chunk - 1);
      marker.id = static_cast<int>(exp_id);
      marker.type = visualization_msgs::msg::Marker::POINTS;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = exp->x_m;
      marker.pose.position.y = exp->y_m;
      marker.pose.position.z = 0.0;
      tf2::Quaternion q_pose;
      q_pose.setRPY(0.0, 0.0, exp->th_rad);
      marker.pose.orientation = tf2::toMsg(q_pose);
      marker.scale.x = marker.scale.y = marker.scale.z = marker_point_size_;
      const float hue = std::fmod(static_cast<float>(exp_id) * 0.6180339887f, 1.0f);
      float r = 1.0f, g = 1.0f, b = 1.0f;
      hsv_to_rgb(hue, 1.0f, 0.95f, r, g, b);
      marker.color.r = r;
      marker.color.g = g;
      marker.color.b = b;
      marker.color.a = 0.85f;
      marker.points.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        marker.points[i].x = cloud.xyz[3 * i + 0];
        marker.points[i].y = cloud.xyz[3 * i + 1];
        marker.points[i].z = cloud.xyz[3 * i + 2];
      }

      visualization_msgs::msg::MarkerArray array;
      array.markers.push_back(std::move(marker));
      pub_keyframe_markers_->publish(array);
      last_marker_pose_[exp_id] = {exp->x_m, exp->y_m, exp->th_rad};
      ++markers_published_;
    }

    /// Overview cloud: every experience rendered into the map frame.
    void publish_global_cloud()
    {
      const double voxel = global_cloud_voxel_size_;
      std::vector<float> buffer;  // x, y, z, rgb (as float bits)
      std::unordered_set<std::uint64_t> occupied;
      const int num_exps = em->get_num_experiences();
      int used = 0;

      const auto cell_key = [](int ix, int iy, int iz) {
        return (static_cast<std::uint64_t>(ix & 0x1FFFFF) << 42) |
               (static_cast<std::uint64_t>(iy & 0x1FFFFF) << 21) |
               static_cast<std::uint64_t>(iz & 0x1FFFFF);
      };

      for (int id = 0; id < num_exps; ++id) {
        Experience * exp = em->get_experience(id);
        if (exp == nullptr) {
          continue;
        }
        const auto it = keyframe_clouds_.find(static_cast<uint32_t>(exp->vt_id));
        if (it == keyframe_clouds_.end()) {
          continue;
        }
        const KeyframeCloud & cloud = it->second;
        const double c = std::cos(exp->th_rad);
        const double s = std::sin(exp->th_rad);
        const float hue = std::fmod(static_cast<float>(id) * 0.6180339887f, 1.0f);
        float r = 1.0f, g = 1.0f, b = 1.0f;
        hsv_to_rgb(hue, 1.0f, 0.95f, r, g, b);
        const std::uint32_t rgb =
          (0u << 24) | (static_cast<std::uint32_t>(r * 255.0f) << 16) |
          (static_cast<std::uint32_t>(g * 255.0f) << 8) |
          static_cast<std::uint32_t>(b * 255.0f);
        float rgb_float = 0.0f;
        std::memcpy(&rgb_float, &rgb, sizeof(float));

        const std::size_t n = cloud.size();
        for (std::size_t i = 0; i < n; ++i) {
          const double px = cloud.xyz[3 * i + 0];
          const double py = cloud.xyz[3 * i + 1];
          const double pz = cloud.xyz[3 * i + 2];
          const double mx = c * px - s * py + exp->x_m;
          const double my = s * px + c * py + exp->y_m;
          if (voxel > 0.0) {
            if (!occupied.insert(cell_key(
                  static_cast<int>(std::floor(mx / voxel)),
                  static_cast<int>(std::floor(my / voxel)),
                  static_cast<int>(std::floor(pz / voxel)))).second) {
              continue;
            }
          }
          buffer.push_back(static_cast<float>(mx));
          buffer.push_back(static_cast<float>(my));
          buffer.push_back(static_cast<float>(pz));
          buffer.push_back(rgb_float);
        }
        ++used;
      }

      sensor_msgs::msg::PointCloud2 out;
      out.header.frame_id = "map";
      out.header.stamp = this->now();
      out.height = 1;
      out.width = static_cast<std::uint32_t>(buffer.size() / 4);
      out.is_bigendian = false;
      out.is_dense = true;
      out.point_step = 16;
      out.row_step = out.point_step * out.width;
      const char * names[4] = {"x", "y", "z", "rgb"};
      for (std::uint32_t i = 0; i < 4; ++i) {
        sensor_msgs::msg::PointField field;
        field.name = names[i];
        field.offset = i * 4;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        out.fields.push_back(field);
      }
      out.data.resize(buffer.size() * sizeof(float));
      std::memcpy(out.data.data(), buffer.data(), out.data.size());
      pub_keyframe_cloud_->publish(out);

      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this, 5000,
        "Keyframe overview cloud: %u points from %d experiences "
        "(%zu clouds stored, %zu markers published, voxel %.2f m)",
        out.width, used, keyframe_clouds_.size(), markers_published_, voxel);
    }

    /// Re-sends the markers whose experience pose changed (relaxation, loops).
    void refresh_changed_markers()
    {
      int changed = 0;
      const int num_exps = em->get_num_experiences();
      for (int id = 0; id < num_exps; ++id) {
        const auto known = last_marker_pose_.find(id);
        if (known == last_marker_pose_.end()) {
          continue;
        }
        Experience * exp = em->get_experience(id);
        if (exp == nullptr) {
          continue;
        }
        const auto & prev = known->second;
        if (std::abs(exp->x_m - prev[0]) > 1e-3 ||
            std::abs(exp->y_m - prev[1]) > 1e-3 ||
            std::abs(exp->th_rad - prev[2]) > 1e-3) {
          publish_experience_marker(static_cast<unsigned int>(id));
          ++changed;
        }
      }
      if (changed > 0) {
        RCLCPP_INFO(this->get_logger(),
                    "Keyframe markers refreshed for %d corrected experiences", changed);
      }
    }

    void cloud_timer_callback()
    {
      const rclcpp::Time now = this->now();
      if (clouds_dirty_ && (now - last_cloud_publish_).seconds() >= cloud_publish_period_) {
        publish_global_cloud();
        last_cloud_publish_ = now;
        clouds_dirty_ = false;
      }
      if (marker_refresh_period_ > 0.0 &&
          (now - last_marker_refresh_).seconds() >= marker_refresh_period_) {
        refresh_changed_markers();
        last_marker_refresh_ = now;
      }
    }

    // Publishers
    rclcpp::Publisher<topological_msgs::msg::TopologicalMap>::SharedPtr pub_em;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_em_markers;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_goal_path;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rviz_markers_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_nav_marker_;

    
    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry;
    rclcpp::Subscription<topological_msgs::msg::TopologicalAction>::SharedPtr sub_action;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal;
    
    // Timers and counters
    rclcpp::Time prev_time{0, 0, RCL_ROS_TIME};
    rclcpp::Time prev_goal_update{0, 0, RCL_ROS_TIME};
    rclcpp::Time prev_pub_time{0, 0, RCL_ROS_TIME};
    int action_counter = 0;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ---- Keyframe clouds ----
    struct KeyframeCloud
    {
      std::vector<float> xyz;   // base_frame, 3 floats per point
      std::size_t size() const { return xyz.size() / 3; }
    };

    bool store_keyframe_clouds_ = true;
    std::string base_frame_ = "base_link";
    std::array<double, 3> sensor_mount_xyz_{{0.0, 0.0, 0.3}};
    std::array<double, 3> sensor_mount_rpy_{{0.0, 0.0, 0.0}};
    double global_cloud_voxel_size_ = 0.30;
    double cloud_publish_period_ = 5.0;
    int marker_namespace_chunk_ = 100;
    double marker_refresh_period_ = 30.0;
    double marker_point_size_ = 0.025;
    double z_band_min_ = 0.2;
    double z_band_max_ = 0.4;
    bool mount_logged_ = false;
    std::size_t markers_published_ = 0;

    std::unordered_map<std::uint32_t, KeyframeCloud> keyframe_clouds_;   // by vt_id
    std::unordered_map<std::uint32_t, std::uint32_t> pending_experiences_;  // exp_id -> vt_id
    std::unordered_map<std::uint32_t, std::array<double, 3>> last_marker_pose_;
    bool clouds_dirty_ = false;
    rclcpp::Time last_cloud_publish_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_marker_refresh_{0, 0, RCL_ROS_TIME};

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_keyframe_markers_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_keyframe_cloud_;
    rclcpp::Subscription<topological_msgs::msg::KeyframeCloud>::SharedPtr sub_keyframe_cloud_;
    rclcpp::TimerBase::SharedPtr cloud_timer_;

    // Map Manager
    std::shared_ptr<htmslam::MapManager> map_manager_;
    std::string topic_root_;
    std::shared_ptr<StateManager> state_manager_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_json_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_yaml_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_binary_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_all_service_;
    rclcpp::Service<htmslam::srv::ImportMap>::SharedPtr import_map_service_;
    rclcpp::Service<htmslam::srv::ListMaps>::SharedPtr list_maps_service_;
    rclcpp::Service<htmslam::srv::SetGoal>::SharedPtr set_goal_service_;

    std::string current_mode_ = "mapping";
    bool holonomic_odometry_ = false;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_subscriber_;
    
  #ifdef HAVE_IRRLICHT
    ExperienceMapScene *ems = nullptr;
    bool use_graphics;
  #endif
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ExperienceMapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}