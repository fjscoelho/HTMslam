#ifndef MAP_VISUALIZATION_EXPORTER_HPP
#define MAP_VISUALIZATION_EXPORTER_HPP

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/empty.hpp>
#include <fstream>
#include <json/json.h>
#include "experience_map.h"

namespace htmslam {

class MapVisualizationExporter : public rclcpp::Node
{
public:
    MapVisualizationExporter(ExperienceMap* em_ptr) 
        : Node("map_visualization_exporter"), em_(em_ptr)
    {
        // Publicador para os marcadores RViz
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/experience_map/rviz_markers", 10);
        
        // Serviço para salvar marcadores em arquivo
        save_service_ = this->create_service<std_srvs::srv::Empty>(
            "/experience_map/save_rviz_markers",
            std::bind(&MapVisualizationExporter::save_markers_callback,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2)
        );
        
        // Timer para publicar marcadores periodicamente (a cada 1 segundo)
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MapVisualizationExporter::publish_markers, this)
        );
        
        // Publica imediatamente na inicialização
        publish_markers();
        
        RCLCPP_INFO(this->get_logger(), "Map Visualization Exporter initialized");
        RCLCPP_INFO(this->get_logger(), "  - Publishing markers to: /experience_map/rviz_markers");
        RCLCPP_INFO(this->get_logger(), "  - Call /experience_map/save_rviz_markers to save to file");
    }

private:
    ExperienceMap* em_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr save_service_;
    rclcpp::TimerBase::SharedPtr timer_;

    void publish_markers()
    {
        // SEMPRE publica, mesmo se o mapa estiver vazio (para debug)
        if (em_ == nullptr) {
            RCLCPP_WARN(this->get_logger(), "em_ is null!");
            return;
        }
        
        int num_exps = em_->get_num_experiences();
        int num_links = em_->get_num_links();
        
        // Log a cada 10 publicações para não poluir
        static int pub_count = 0;
        pub_count++;
        if (pub_count % 10 == 0) {
            RCLCPP_INFO(this->get_logger(), "Publishing markers - Experiences: %d, Links: %d", 
                        num_exps, num_links);
        }

        visualization_msgs::msg::MarkerArray marker_array;
        
        // Marcador 1: Nós como esferas
        visualization_msgs::msg::Marker nodes_marker;
        nodes_marker.header.frame_id = "map";
        nodes_marker.header.stamp = this->now();
        nodes_marker.ns = "experience_nodes";
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
        
        // Popula os nós
        for (int i = 0; i < num_exps; i++) {
            Experience* exp = em_->get_experience(i);
            if (exp == nullptr) continue;
            
            geometry_msgs::msg::Point p;
            p.x = exp->x_m;
            p.y = exp->y_m;
            p.z = 0.0;
            nodes_marker.points.push_back(p);
        }
        
        // Se não houver nós, adiciona um ponto de debug na origem
        if (nodes_marker.points.empty()) {
            geometry_msgs::msg::Point p;
            p.x = 0.0;
            p.y = 0.0;
            p.z = 0.0;
            nodes_marker.points.push_back(p);
            RCLCPP_WARN(this->get_logger(), "No experiences, publishing debug point at origin");
        }
        
        marker_array.markers.push_back(nodes_marker);
        
        // Marcador 2: Arestas como linhas
        if (num_links > 0) {
            visualization_msgs::msg::Marker edges_marker;
            edges_marker.header.frame_id = "map";
            edges_marker.header.stamp = this->now();
            edges_marker.ns = "experience_edges";
            edges_marker.id = 1;
            edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            edges_marker.action = visualization_msgs::msg::Marker::ADD;
            edges_marker.scale.x = 0.01;
            edges_marker.color.r = 1.0;
            edges_marker.color.g = 1.0;
            edges_marker.color.b = 1.0;
            edges_marker.color.a = 1.0;
            
            for (int i = 0; i < num_links; i++) {
                Link* link = em_->get_link(i);
                if (link == nullptr) continue;
                
                Experience* from = em_->get_experience(link->exp_from_id);
                Experience* to = em_->get_experience(link->exp_to_id);
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
            marker_array.markers.push_back(edges_marker);
        }
        
        // Marcador 3: Nó atual
        if (num_exps > 0) {
            visualization_msgs::msg::Marker current_node;
            current_node.header.frame_id = "map";
            current_node.header.stamp = this->now();
            current_node.ns = "current_experience";
            current_node.id = 2;
            current_node.type = visualization_msgs::msg::Marker::SPHERE;
            current_node.action = visualization_msgs::msg::Marker::ADD;
            current_node.scale.x = 0.15;
            current_node.scale.y = 0.15;
            current_node.scale.z = 0.15;
            current_node.color.r = 1.0;
            current_node.color.g = 0.0;
            current_node.color.b = 0.0;
            current_node.color.a = 1.0;
            
            Experience* current = em_->get_experience(em_->get_current_id());
            if (current != nullptr) {
                current_node.pose.position.x = current->x_m;
                current_node.pose.position.y = current->y_m;
                current_node.pose.position.z = 0.0;
                marker_array.markers.push_back(current_node);
            }
        }
        
        // Publica
        marker_pub_->publish(marker_array);
        
        if (pub_count % 10 == 0) {
            RCLCPP_INFO(this->get_logger(), "Published %zu markers", marker_array.markers.size());
        }
    }

    void save_markers_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (em_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "ExperienceMap is null!");
            return;
        }

        std::string filename = "topological_map_rviz_" + 
                              std::to_string(this->now().seconds()) + 
                              ".json";
        
        Json::Value root;
        root["type"] = "topological_map";
        root["timestamp"] = this->now().seconds();
        
        Json::Value nodes_array(Json::arrayValue);
        Json::Value edges_array(Json::arrayValue);
        
        for (int i = 0; i < em_->get_num_experiences(); i++) {
            Experience* exp = em_->get_experience(i);
            if (exp == nullptr) continue;
            
            Json::Value node;
            node["id"] = exp->id;
            node["x"] = exp->x_m;
            node["y"] = exp->y_m;
            node["theta"] = exp->th_rad;
            nodes_array.append(node);
        }
        root["nodes"] = nodes_array;
        
        for (int i = 0; i < em_->get_num_links(); i++) {
            Link* link = em_->get_link(i);
            if (link == nullptr) continue;
            
            Json::Value edge;
            edge["from"] = link->exp_from_id;
            edge["to"] = link->exp_to_id;
            edge["distance"] = link->d;
            edges_array.append(edge);
        }
        root["edges"] = edges_array;
        
        std::ofstream file(filename);
        if (file.is_open()) {
            file << root.toStyledString();
            file.close();
            RCLCPP_INFO(this->get_logger(), "RViz markers saved to: %s", filename.c_str());
            RCLCPP_INFO(this->get_logger(), "  - Nodes: %d", em_->get_num_experiences());
            RCLCPP_INFO(this->get_logger(), "  - Edges: %d", em_->get_num_links());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to save RViz markers to: %s", filename.c_str());
        }
    }
};

} // namespace htmslam

#endif