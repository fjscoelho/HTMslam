#ifndef MAP_MANAGER_HPP
#define MAP_MANAGER_HPP

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <map>

#include <json/json.h>
#include <yaml-cpp/yaml.h>

#include "experience_map.h"

namespace htmslam {

struct MapMetadata {
    std::string version = "1.0";
    std::string timestamp;
    std::string description = "NeoSLAM Topological Map";
    int num_experiences = 0;
    int num_links = 0;
    double exp_correction = 0.5;
    int exp_loops = 10;
    double exp_initial_em_deg = 90.0;
    std::map<std::string, std::string> custom_params;
    // Novo: Informações sobre vt_id
    bool has_vt_id = true;  // Indica que o mapa tem vt_id
    int vt_id_start = 0;    // Valor inicial do vt_id
    
    MapMetadata() {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S");
        timestamp = ss.str();
    }
};

class MapManager {
public:
    MapManager(ExperienceMap* em, const std::string& states_dir = "./htmslam_exported_states/");
    ~MapManager() = default;
    
    // Exportação
    bool export_json(const std::string& filename, const MapMetadata& metadata = MapMetadata());
    bool export_yaml(const std::string& filename, const MapMetadata& metadata = MapMetadata());
    bool export_binary(const std::string& filename, const MapMetadata& metadata = MapMetadata());
    bool export_all(const std::string& base_name, const MapMetadata& metadata = MapMetadata());
    
    // Importação
    bool import_map(const std::string& filename);
    
    // Gerenciamento
    std::vector<std::string> list_maps() const;
    bool delete_map(const std::string& filename);
    std::string get_states_dir() const { return states_dir_; }
    
private:
    ExperienceMap* em_;
    std::string states_dir_;
    
    // Serialização
    Json::Value serialize_to_json(const MapMetadata& metadata);
    YAML::Node serialize_to_yaml(const MapMetadata& metadata);
    std::vector<uint8_t> serialize_to_binary(const MapMetadata& metadata);
    
    // Deserialização
    bool load_from_json(const std::string& filename);
    bool load_from_yaml(const std::string& filename);
    bool load_from_binary(const std::string& filename);
    
    // Helpers
    void create_directory();
    std::string get_timestamp() const;
    bool validate_metadata(const MapMetadata& metadata);
};

} // namespace htmslam

#endif