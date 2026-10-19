#ifndef EXPORT_MAP_SERVICE_HPP
#define EXPORT_MAP_SERVICE_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#include <fstream>
#include <json/json.h>
#include "experience_map.h"

namespace htmslam {

class ExportMapService : public rclcpp::Node
{
public:
    ExportMapService(ExperienceMap* em_ptr) 
        : Node("export_map_service"), em_(em_ptr)
    {
        // Serviço para exportação simples
        export_service_ = this->create_service<std_srvs::srv::Empty>(
            "/experience_map/export_graph",
            std::bind(&ExportMapService::export_graph_callback, 
                      this, 
                      std::placeholders::_1, 
                      std::placeholders::_2)
        );
        
        // Serviço para exportação com nome customizado
        export_with_name_service_ = this->create_service<std_srvs::srv::Empty>(
            "/experience_map/export_graph_named",
            std::bind(&ExportMapService::export_graph_named_callback,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2)
        );
        
        RCLCPP_INFO(this->get_logger(), "Export Map Service initialized");
    }

private:
    ExperienceMap* em_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr export_with_name_service_;

    void export_graph_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (em_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "ExperienceMap is null!");
            return;
        }
        
        std::string default_filename = "topological_map_" + 
                                       std::to_string(this->now().seconds()) + 
                                       ".json";
        export_graph_to_json(default_filename);
    }

    void export_graph_named_callback(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (em_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "ExperienceMap is null!");
            return;
        }
        
        // Tenta obter o parâmetro do nó pai
        std::string filename;
        auto parent_node = std::dynamic_pointer_cast<rclcpp::Node>(this->shared_from_this());
        if (parent_node) {
            // Busca o parâmetro no nó pai (experience_map_node)
            try {
                filename = parent_node->get_parameter("export_filename").as_string();
            } catch (const rclcpp::exceptions::ParameterNotDeclaredException& e) {
                RCLCPP_WARN(this->get_logger(), "Parameter 'export_filename' not found, using default");
                filename = "topological_map_" + std::to_string(this->now().seconds()) + ".json";
            }
        } else {
            filename = "topological_map_" + std::to_string(this->now().seconds()) + ".json";
        }
        
        export_graph_to_json(filename);
    }

    void export_graph_to_json(const std::string& filename)
    {
        if (em_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "ExperienceMap is null!");
            return;
        }

        Json::Value root;
        
        root["metadata"]["timestamp"] = this->now().seconds();
        root["metadata"]["num_experiences"] = em_->get_num_experiences();
        root["metadata"]["num_links"] = em_->get_num_links();
        root["metadata"]["current_experience_id"] = em_->get_current_id();
        
        Json::Value nodes_array(Json::arrayValue);
        for (int i = 0; i < em_->get_num_experiences(); i++) {
            Experience* exp = em_->get_experience(i);
            if (exp == nullptr) continue;
            
            Json::Value node;
            node["id"] = exp->id;
            node["x_m"] = exp->x_m;
            node["y_m"] = exp->y_m;
            node["th_rad"] = exp->th_rad;
            node["seconds"] = exp->seconds;
            node["nanoseconds"] = exp->nanoseconds;
            nodes_array.append(node);
        }
        root["nodes"] = nodes_array;
        
        Json::Value edges_array(Json::arrayValue);
        for (int i = 0; i < em_->get_num_links(); i++) {
            Link* link = em_->get_link(i);
            if (link == nullptr) continue;
            
            Json::Value edge;
            edge["id"] = i;
            edge["exp_from_id"] = link->exp_from_id;
            edge["exp_to_id"] = link->exp_to_id;
            edge["d"] = link->d;
            edge["heading_rad"] = link->heading_rad;
            edge["facing_rad"] = link->facing_rad;
            edge["delta_time_s"] = link->delta_time_s;
            edges_array.append(edge);
        }
        root["edges"] = edges_array;
        
        std::ofstream file(filename);
        if (file.is_open()) {
            file << root.toStyledString();
            file.close();
            RCLCPP_INFO(this->get_logger(), "Map exported successfully to: %s", filename.c_str());
            RCLCPP_INFO(this->get_logger(), "  - Nodes: %d", em_->get_num_experiences());
            RCLCPP_INFO(this->get_logger(), "  - Edges: %d", em_->get_num_links());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", filename.c_str());
        }
    }
};

} // namespace htmslam

#endif // EXPORT_MAP_SERVICE_HPP