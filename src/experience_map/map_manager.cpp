#include "map_manager.hpp"
#include <iostream>
#include <cstring>

namespace htmslam 
{

    MapManager::MapManager(ExperienceMap* em, const std::string& states_dir) 
        : em_(em), states_dir_(states_dir) {
        create_directory();
    }

    void MapManager::create_directory() {
        if (!std::filesystem::exists(states_dir_)) {
            std::filesystem::create_directories(states_dir_);
            std::cout << "Created map directory: " << states_dir_ << std::endl;
        }
    }

    std::string MapManager::get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");
        return ss.str();
    }

    // ============================================
    // SERIALIZATION TO JSON
    // ============================================

    Json::Value MapManager::serialize_to_json(const MapMetadata& metadata) {
        Json::Value root;
        
        // Metadados
        root["metadata"]["version"] = metadata.version;
        root["metadata"]["timestamp"] = metadata.timestamp;
        root["metadata"]["description"] = metadata.description;
        root["metadata"]["num_experiences"] = em_->get_num_experiences();
        root["metadata"]["num_links"] = em_->get_num_links();
        root["metadata"]["current_experience_id"] = em_->get_current_id();
        root["metadata"]["exp_correction"] = metadata.exp_correction;
        root["metadata"]["exp_loops"] = metadata.exp_loops;
        root["metadata"]["exp_initial_em_deg"] = metadata.exp_initial_em_deg;
        root["metadata"]["has_vt_id"] = metadata.has_vt_id;  // <-- ADICIONADO
        
        // Custom parameters
        for (const auto& param : metadata.custom_params) {
            root["metadata"]["custom_params"][param.first] = param.second;
        }
        
        // Nodes (Experiences)
        Json::Value nodes_array(Json::arrayValue);
        for (int i = 0; i < em_->get_num_experiences(); i++) {
            Experience* exp = em_->get_experience(i);
            if (exp == nullptr) continue;
            
            Json::Value node;
            node["id"] = exp->id;
            node["vt_id"] = exp->vt_id;  // <-- ADICIONADO
            node["x_m"] = exp->x_m;
            node["y_m"] = exp->y_m;
            node["th_rad"] = exp->th_rad;
            node["seconds"] = exp->seconds;
            node["nanoseconds"] = exp->nanoseconds;
            
            // List of links from and to this experience
            Json::Value links_from(Json::arrayValue);
            for (unsigned int link_id : exp->links_from) {
                links_from.append(link_id);
            }
            node["links_from"] = links_from;
            
            Json::Value links_to(Json::arrayValue);
            for (unsigned int link_id : exp->links_to) {
                links_to.append(link_id);
            }
            node["links_to"] = links_to;
            
            nodes_array.append(node);
        }
        root["nodes"] = nodes_array;
        
        // Edges (Links)
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
        Json::Value state;
            state["current_exp_id"] = em_->get_current_id();
            const auto& goals = em_->get_goals();
            for (int g : goals) {
                state["goal_list"].append(g);
            }
            state["relative_rad"] = em_->getRelativeRad();  // precisa tornar acessível
            root["state"] = state;

        
        return root;
    }

    bool MapManager::export_json(const std::string& filename, const MapMetadata& metadata) {
        Json::Value root = serialize_to_json(metadata);
        
        std::string full_path = states_dir_ + filename;
        if (filename.find(".json") == std::string::npos) {
            full_path += ".json";
        }
        
        std::ofstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << full_path << std::endl;
            return false;
        }
        
        file << root.toStyledString();
        file.close();
        
        std::cout << "JSON map exported to: " << full_path << std::endl;
        std::cout << "  - Nodes: " << em_->get_num_experiences() << std::endl;
        std::cout << "  - Links: " << em_->get_num_links() << std::endl;
        
        return true;
    }

    // ============================================
    // SERIALIZATION TO YAML
    // ============================================

    YAML::Node MapManager::serialize_to_yaml(const MapMetadata& metadata) {
        YAML::Node root;
        
        // Metadados
        root["metadata"]["version"] = metadata.version;
        root["metadata"]["timestamp"] = metadata.timestamp;
        root["metadata"]["description"] = metadata.description;
        root["metadata"]["num_experiences"] = em_->get_num_experiences();
        root["metadata"]["num_links"] = em_->get_num_links();
        root["metadata"]["current_experience_id"] = em_->get_current_id();
        root["metadata"]["exp_correction"] = metadata.exp_correction;
        root["metadata"]["exp_loops"] = metadata.exp_loops;
        root["metadata"]["exp_initial_em_deg"] = metadata.exp_initial_em_deg;
        root["metadata"]["has_vt_id"] = metadata.has_vt_id;  // <-- ADICIONADO
        
        // Nodes (Experiences)
        for (int i = 0; i < em_->get_num_experiences(); i++) {
            Experience* exp = em_->get_experience(i);
            if (exp == nullptr) continue;
            
            YAML::Node node;
            node["id"] = exp->id;
            node["vt_id"] = exp->vt_id;  // <-- ADICIONADO
            node["x_m"] = exp->x_m;
            node["y_m"] = exp->y_m;
            node["th_rad"] = exp->th_rad;
            node["seconds"] = exp->seconds;
            node["nanoseconds"] = exp->nanoseconds;
            
        
            root["nodes"].push_back(node);
        }
        
        // Edges (Links)
        for (int i = 0; i < em_->get_num_links(); i++) {
            Link* link = em_->get_link(i);
            if (link == nullptr) continue;
            
            YAML::Node edge;
            edge["id"] = i;
            edge["exp_from_id"] = link->exp_from_id;
            edge["exp_to_id"] = link->exp_to_id;
            edge["d"] = link->d;
            edge["heading_rad"] = link->heading_rad;
            edge["facing_rad"] = link->facing_rad;
            edge["delta_time_s"] = link->delta_time_s;
            
            root["edges"].push_back(edge);
        }
        
        return root;
    }

    bool MapManager::export_yaml(const std::string& filename, const MapMetadata& metadata) {
        YAML::Node root = serialize_to_yaml(metadata);
        
        std::string full_path = states_dir_ + filename;
        if (filename.find(".yaml") == std::string::npos && 
            filename.find(".yml") == std::string::npos) {
            full_path += ".yaml";
        }
        
        std::ofstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << full_path << std::endl;
            return false;
        }
        
        file << root;
        file.close();
        
        std::cout << "YAML map exported to: " << full_path << std::endl;
        std::cout << "  - Nodes: " << em_->get_num_experiences() << std::endl;
        std::cout << "  - Links: " << em_->get_num_links() << std::endl;
        
        return true;
    }

    // ============================================
    // SERIALIZATION TO BINARY
    // ============================================

    std::vector<uint8_t> MapManager::serialize_to_binary(const MapMetadata& metadata) {
        std::vector<uint8_t> buffer;
        
        // Helper to write to buffer
        auto write_to_buffer = [&buffer](const void* data, size_t size) {
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            buffer.insert(buffer.end(), bytes, bytes + size);
        };
        
        // Header (magic number + version)
        uint32_t magic = 0x4E454F53; // "NEOS" in hex
        uint32_t version = 2;  // <-- INCREMENT VERSION (2 = with vt_id)
        write_to_buffer(&magic, sizeof(magic));
        write_to_buffer(&version, sizeof(version));
        
        // Basic metadata
        int num_experiences = em_->get_num_experiences();
        int num_links = em_->get_num_links();
        int current_id = em_->get_current_id();
        double exp_correction = metadata.exp_correction;
        int exp_loops = metadata.exp_loops;
        double exp_initial_em_deg = metadata.exp_initial_em_deg;
        
        write_to_buffer(&num_experiences, sizeof(num_experiences));
        write_to_buffer(&num_links, sizeof(num_links));
        write_to_buffer(&current_id, sizeof(current_id));
        write_to_buffer(&exp_correction, sizeof(exp_correction));
        write_to_buffer(&exp_loops, sizeof(exp_loops));
        write_to_buffer(&exp_initial_em_deg, sizeof(exp_initial_em_deg));
        
        // Nodes (Experiences)
        for (int i = 0; i < num_experiences; i++) {
            Experience* exp = em_->get_experience(i);
            if (exp == nullptr) continue;
            
            write_to_buffer(&exp->id, sizeof(exp->id));
            write_to_buffer(&exp->vt_id, sizeof(exp->vt_id));  // <-- ADICIONADO
            write_to_buffer(&exp->x_m, sizeof(exp->x_m));
            write_to_buffer(&exp->y_m, sizeof(exp->y_m));
            write_to_buffer(&exp->th_rad, sizeof(exp->th_rad));
            write_to_buffer(&exp->seconds, sizeof(exp->seconds));
            write_to_buffer(&exp->nanoseconds, sizeof(exp->nanoseconds));
            
            // Quantity of links
            int links_from_size = exp->links_from.size();
            int links_to_size = exp->links_to.size();
            write_to_buffer(&links_from_size, sizeof(links_from_size));
            write_to_buffer(&links_to_size, sizeof(links_to_size));
            
            // Links
            for (unsigned int link_id : exp->links_from) {
                write_to_buffer(&link_id, sizeof(link_id));
            }
            for (unsigned int link_id : exp->links_to) {
                write_to_buffer(&link_id, sizeof(link_id));
            }
        }
        
        // Edges (Links)
        for (int i = 0; i < num_links; i++) {
            Link* link = em_->get_link(i);
            if (link == nullptr) continue;
            
            write_to_buffer(&link->exp_from_id, sizeof(link->exp_from_id));
            write_to_buffer(&link->exp_to_id, sizeof(link->exp_to_id));
            write_to_buffer(&link->d, sizeof(link->d));
            write_to_buffer(&link->heading_rad, sizeof(link->heading_rad));
            write_to_buffer(&link->facing_rad, sizeof(link->facing_rad));
            write_to_buffer(&link->delta_time_s, sizeof(link->delta_time_s));
        }
        
        return buffer;
    }

    bool MapManager::export_binary(const std::string& filename, const MapMetadata& metadata) {
        std::vector<uint8_t> buffer = serialize_to_binary(metadata);
        
        std::string full_path = states_dir_ + filename;
        if (filename.find(".bin") == std::string::npos) {
            full_path += ".bin";
        }
        
        std::ofstream file(full_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << full_path << std::endl;
            return false;
        }
        
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        file.close();
        
        std::cout << "Binary map exported to: " << full_path << std::endl;
        std::cout << "  - Size: " << buffer.size() << " bytes" << std::endl;
        std::cout << "  - Nodes: " << em_->get_num_experiences() << std::endl;
        std::cout << "  - Links: " << em_->get_num_links() << std::endl;
        
        return true;
    }

    // ============================================
    // EXPORTATION TO ALL FORMATS
    // ============================================

    bool MapManager::export_all(const std::string& base_name, const MapMetadata& metadata) {
        bool success = true;
        
        // JSON
        success &= export_json(base_name + ".json", metadata);
        
        // YAML
        success &= export_yaml(base_name + ".yaml", metadata);
        
        // Binary
        success &= export_binary(base_name + ".bin", metadata);
        
        if (success) {
            std::cout << "All formats exported successfully!" << std::endl;
            std::cout << "Base name: " << base_name << std::endl;
            std::cout << "Location: " << states_dir_ << std::endl;
        }
        
        return success;
    }

    // ============================================
    // IMPORTATION
    // ============================================

    bool MapManager::import_map(const std::string& filename) {
        std::string full_path = states_dir_ + filename;
        
        // Detectar formato pela extensão
        if (filename.find(".json") != std::string::npos) {
            return load_from_json(full_path);
        } else if (filename.find(".yaml") != std::string::npos || 
                filename.find(".yml") != std::string::npos) {
            return load_from_yaml(full_path);
        } else if (filename.find(".bin") != std::string::npos) {
            return load_from_binary(full_path);
        } else {
            std::cerr << "Unknown file format: " << filename << std::endl;
            return false;
        }
    }

    bool MapManager::load_from_json(const std::string& filename) {
        Json::Value root;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        
        file >> root;
        file.close();
        
        // Verificar metadados
        if (!root.isMember("metadata")) {
            std::cerr << "Invalid map file: missing metadata" << std::endl;
            return false;
        }
        
        // Verificar versão
        std::string version = root["metadata"]["version"].asString();
        if (version != "1.0" && version != "1.1") {
            std::cerr << "Unsupported map version: " << version << std::endl;
            return false;
        }
        
        bool has_vt_id = version == "1.1" || root["metadata"]["has_vt_id"].asBool();
        
        std::cout << "Loading map from JSON: " << filename << std::endl;
        std::cout << "  - Version: " << version << std::endl;
        std::cout << "  - Has vt_id: " << (has_vt_id ? "yes" : "no") << std::endl;
        std::cout << "  - Nodes: " << root["nodes"].size() << std::endl;
        std::cout << "  - Links: " << root["edges"].size() << std::endl;
        
        // Limpar o mapa atual
        if (em_ != nullptr) {
            em_->clear();
            std::cout << "  - Cleared current map" << std::endl;
        }
        
        // ============================================
        // 1. Add nodes (Experiences)
        // ============================================
        const Json::Value& nodes = root["nodes"];
        for (const auto& node_json : nodes) {
            int id = node_json["id"].asInt();
            int vt_id = has_vt_id ? node_json["vt_id"].asInt() : -1;
            double x = node_json["x_m"].asDouble();
            double y = node_json["y_m"].asDouble();
            double th = node_json["th_rad"].asDouble();
            unsigned int sec = node_json["seconds"].asUInt();
            unsigned int nsec = node_json["nanoseconds"].asUInt();
            
            em_->add_experience_from_import(id, vt_id, x, y, th, sec, nsec);
        }
        std::cout << "  - Added " << nodes.size() << " experiences" << std::endl;
        
        // ============================================
        // 2. Add links (Edges)
        // ============================================
        const Json::Value& edges = root["edges"];
        for (const auto& edge_json : edges) {
            int id = edge_json["id"].asInt();
            int from = edge_json["exp_from_id"].asInt();
            int to = edge_json["exp_to_id"].asInt();
            double d = edge_json["d"].asDouble();
            double heading = edge_json["heading_rad"].asDouble();
            double facing = edge_json["facing_rad"].asDouble();
            double delta_time = edge_json["delta_time_s"].asDouble();
            
            em_->add_link_from_import(id, from, to, d, heading, facing, delta_time);
        }
        std::cout << "  - Added " << edges.size() << " links" << std::endl;
        
        // ============================================
        // 3. Restore map state (current_exp_id, goals, etc.)
        // ============================================
        if (root.isMember("state")) {
            const Json::Value& state = root["state"];
            if (state.isMember("current_exp_id")) {
                int current_id = state["current_exp_id"].asInt();
                // Usar on_set_experience para atualizar o estado (respeitando o modo)
                em_->on_set_experience(current_id, 0.0);
            }
            if (state.isMember("goal_list")) {
                em_->clear_goal_list();
                for (const auto& goal_id : state["goal_list"]) {
                    em_->add_goal(goal_id.asInt());
                }
            }
            if (state.isMember("relative_rad")) {
                // Ajuste adicional se necessário
            }
            std::cout << "  - Restored state: current_exp_id=" << em_->get_current_id()
                    << ", goals=" << em_->get_goals().size() << std::endl;
        }
        
        std::cout << "Map imported successfully!" << std::endl;
        std::cout << "  - Nodes: " << em_->get_num_experiences() << std::endl;
        std::cout << "  - Links: " << em_->get_num_links() << std::endl;
        
        return true;
    }

    bool MapManager::load_from_yaml(const std::string& filename) {
        try {
            YAML::Node root = YAML::LoadFile(filename);
            
            // Verify metadata
            if (!root["metadata"]) {
                std::cerr << "Invalid YAML: missing metadata" << std::endl;
                return false;
            }
            
            std::cout << "Loading map from YAML: " << filename << std::endl;
            std::cout << "  - Nodes: " << root["nodes"].size() << std::endl;
            std::cout << "  - Links: " << root["edges"].size() << std::endl;
            
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML parsing error: " << e.what() << std::endl;
            return false;
        }
    }

    bool MapManager::load_from_binary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> buffer(file_size);
        file.read(reinterpret_cast<char*>(buffer.data()), file_size);
        file.close();
        
        // TODO: Deserialize the binary data into the ExperienceMap (em_)
        std::cout << "Loading map from binary: " << filename << std::endl;
        std::cout << "  - Size: " << file_size << " bytes" << std::endl;
        
        return true;
    }

    // ============================================
    // MAP MANAGEMENT
    // ============================================

    std::vector<std::string> MapManager::list_maps() const {
        std::vector<std::string> maps;
        
        if (!std::filesystem::exists(states_dir_)) {
            return maps;
        }
        
        for (const auto& entry : std::filesystem::directory_iterator(states_dir_)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                // Filtrar apenas arquivos de mapa
                if (path.find(".json") != std::string::npos ||
                    path.find(".yaml") != std::string::npos ||
                    path.find(".yml") != std::string::npos ||
                    path.find(".bin") != std::string::npos) {
                    maps.push_back(entry.path().filename().string());
                }
            }
        }
        
        return maps;
    }

    bool MapManager::delete_map(const std::string& filename) {
        std::string full_path = states_dir_ + filename;
        if (!std::filesystem::exists(full_path)) {
            std::cerr << "File does not exist: " << filename << std::endl;
            return false;
        }
        
        try {
            std::filesystem::remove(full_path);
            std::cout << "Deleted map: " << filename << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to delete map: " << e.what() << std::endl;
            return false;
        }
    }

} // namespace htmslam