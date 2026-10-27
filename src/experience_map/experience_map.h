#ifndef _EXPERIENCE_MAP_H_
#define _EXPERIENCE_MAP_H_

#define _USE_MATH_DEFINES
#include "math.h"
#include "utils.h"

#include <stdio.h>
#include <vector>
#include <deque>

#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>

/*
 * The Link structure describes a link
 * between two experiences.
 */

struct Link
{
  double d;
  double heading_rad;
  double facing_rad;
  int exp_to_id;
  int exp_from_id;
  double delta_time_s;

  template<typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & d;
      ar & heading_rad;
      ar & facing_rad;
      ar & exp_to_id;
      ar & exp_from_id;
      ar & delta_time_s;
    }

};

/*
 * The Experience structure describes
 * a node in the Experience_Map.
 */
struct Experience
{
  int id; // its own id

  double x_m, y_m, th_rad;
  int vt_id;
  unsigned int  seconds;
  unsigned int nanoseconds;
  std::vector<unsigned int> links_from; // links from this experience
  std::vector<unsigned int> links_to; // links to this experience


  // goal navigation
  double time_from_current_s;
  unsigned int goal_to_current, current_to_goal;

  template<typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & id;
      ar & x_m & y_m & th_rad;
      ar & vt_id;
      ar & seconds;
      ar & nanoseconds;
      ar & links_from & links_to;
      ar & time_from_current_s;
      ar & goal_to_current & current_to_goal;
    }

};

class ExperienceMapScene;

class ExperienceMap
{

public:
  friend class ExperienceMapScene;

  ExperienceMap(double exp_correction, int exp_loops, double exp_initial_em_deg);
  ~ExperienceMap();

  // create a new experience for a given position
  int on_create_experience(unsigned int exp_id, unsigned int seconds, unsigned int nanoseconds, unsigned int vt_id);
  bool on_create_link(int exp_id_from, int exp_id_to, double rel_rad);

  Experience *get_experience(int id)
  {
    return &experiences[id];
  }

  const Experience *get_experience(int id) const
  {
    return &experiences[id];
  }
  
  Link * get_link(int id)
  {
    return &links[id];
  }

  // update the current position of the experience map
  // since the last experience
  void on_odo(double vtrans, double vrot, double time_diff_s);

  // Same as on_odo, but the translation direction is decoupled from the
  // heading (holonomic robots). `vtrans` is the (non-negative) speed and
  // `dheading` the direction of the translation in the body frame (rad).
  void on_odo_holonomic(double vtrans, double dheading, double vrot, double time_diff_s);

  // update the map by relaxing the graph
  bool iterate();

  // change the current experience
  int on_set_experience(int new_exp_id, double rel_rad);

  int get_num_experiences()
  {
    return experiences.size();
  }

  int get_num_links()
  {
    return links.size();
  }

  int get_current_id()
  {
    return current_exp_id;
  }

  // functions for setting and handling goals.
  void add_goal(double x_m, double y_m);
  void add_goal(int id)
  {
    goal_list.push_back(id);
  }
  bool calculate_path_to_goal(double time_s);
  bool get_goal_waypoint();
  void clear_goal_list()
  {
    goal_list.clear();
  }
  int get_current_goal_id()
  {
    return (goal_list.size() == 0) ? -1 : (int)goal_list.front();
  }
  void delete_current_goal()
  {
    goal_list.pop_front();
  }
  bool get_goal_success()
  {
    return goal_success;
  }
  double get_subgoal_m() const;
  double get_subgoal_rad() const;

  const std::deque<int> &get_goals() const
  {
    return goal_list;
  }

  unsigned int get_goal_path_final_exp()
  {
          return goal_path_final_exp_id;
  }

  // Método para limpar o mapa
    void clear() {
        experiences.clear();
        links.clear();
        goal_list.clear();
        current_exp_id = 0;
        prev_exp_id = 0;
        waypoint_exp_id = -1;
        goal_timeout_s = 0;
        goal_success = false;
        accum_delta_x = 0;
        accum_delta_y = 0;
        accum_delta_facing = 0;
        accum_delta_time_s = 0;
    }
    
    // Método para adicionar experiência importada
    void add_experience_from_import(int id, int vt_id, double x, double y, 
                                   double th, unsigned int sec, unsigned int nsec) {
        // Verificar se o ID já existe
        if (id >= 0 && id < (int)experiences.size()) {
            // Atualizar experiência existente
            experiences[id].vt_id = vt_id;
            experiences[id].x_m = x;
            experiences[id].y_m = y;
            experiences[id].th_rad = th;
            experiences[id].seconds = sec;
            experiences[id].nanoseconds = nsec;
        } else {
            // Adicionar nova experiência
            experiences.resize(id + 1);
            Experience* exp = &experiences[id];
            exp->id = id;
            exp->vt_id = vt_id;
            exp->x_m = x;
            exp->y_m = y;
            exp->th_rad = th;
            exp->seconds = sec;
            exp->nanoseconds = nsec;
            exp->goal_to_current = -1;
            exp->current_to_goal = -1;
        }
    }
    
    // Método para adicionar link importado
    void add_link_from_import(int id, int from_id, int to_id, double d, 
                             double heading, double facing, double delta_time) {
        // Verificar se o link já existe
        if (id >= 0 && id < (int)links.size()) {
            // Atualizar link existente
            links[id].exp_from_id = from_id;
            links[id].exp_to_id = to_id;
            links[id].d = d;
            links[id].heading_rad = heading;
            links[id].facing_rad = facing;
            links[id].delta_time_s = delta_time;
        } else {
            // Adicionar novo link
            links.resize(id + 1);
            Link* link = &links[id];
            link->exp_from_id = from_id;
            link->exp_to_id = to_id;
            link->d = d;
            link->heading_rad = heading;
            link->facing_rad = facing;
            link->delta_time_s = delta_time;
        }
        
        // Atualizar as listas de links das experiências
        if (from_id >= 0 && from_id < (int)experiences.size()) {
            experiences[from_id].links_from.push_back(id);
        }
        if (to_id >= 0 && to_id < (int)experiences.size()) {
            experiences[to_id].links_to.push_back(id);
        }
    }

    /**
     * @brief Atualiza a pose odométrica (usada em NAVIGATION)
     * @param vtrans Velocidade linear (m/s)
     * @param vrot Velocidade angular (rad/s)
     * @param time_diff_s Diferença de tempo (s)
     */
    void updateOdomPose(double vtrans, double vrot, double time_diff_s) {
      vtrans = vtrans * time_diff_s;
      vrot = vrot * time_diff_s;
      // Deslocamento no referencial do inicio do intervalo (ver on_odo).
      odom_x_ += vtrans * cos(odom_th_);
      odom_y_ += vtrans * sin(odom_th_);
      odom_th_ = clip_rad_180(odom_th_ + vrot);
      use_odom_pose_ = true;
    }

    /**
     * @brief Reseta a pose odométrica para uma posição específica
     * @param x Posição x (m)
     * @param y Posição y (m)
     * @param th Orientação (rad)
     */
      void resetOdomPose(double x, double y, double th) {
        odom_x_ = x;
        odom_y_ = y;
        odom_th_ = th;
        use_odom_pose_ = true;
      }
    
    /**
     * @brief Retorna à pose do Experience Map (modo MAPPING)
     */
      void useExperiencePose() {
        use_odom_pose_ = false;
      }

      /**
   * @brief Obtém a pose atual do robô (dependendo do modo)
   * @return std::tuple<double, double, double> (x, y, theta)
   */
      std::tuple<double, double, double> getCurrentPose() const {
        if (use_odom_pose_) {
          return std::make_tuple(odom_x_, odom_y_, odom_th_);
        } else {
          // Usa a pose da experiência atual
          const Experience* exp = get_experience(current_exp_id);
          if (exp != nullptr) {
            return std::make_tuple(exp->x_m, exp->y_m, clip_rad_180(exp->th_rad + relative_rad));
          }
          return std::make_tuple(0.0, 0.0, 0.0);
        }
      }
      
      /**
       * @brief Obtém a pose da odometria (x, y, theta)
       */
      std::tuple<double, double, double> getOdomPose() const {
        return std::make_tuple(odom_x_, odom_y_, odom_th_);
      }
      
      /**
       * @brief Verifica se está usando pose da odometria
       */
      bool isUsingOdomPose() const { return use_odom_pose_; }

    /**
     * @brief Corrige a pose do robô para a pose de uma experiência existente
     * @param exp_id ID da experiência destino
     * @param rel_rad Orientação relativa (radianos)
     * @return true se a correção foi aplicada com sucesso
     */
    bool correctPoseToExperience(int exp_id, double rel_rad = 0.0) {
      if (exp_id < 0 || exp_id >= static_cast<int>(experiences.size())) {
        return false;
      }
      
      const Experience& exp = experiences[exp_id];
      
      // Corrige a pose da odometria
      odom_x_ = exp.x_m;
      odom_y_ = exp.y_m;
      odom_th_ = clip_rad_180(exp.th_rad + rel_rad);
      use_odom_pose_ = true;
      
      // Atualiza também a experiência atual (opcional, para navegação)
      current_exp_id = exp_id;
      relative_rad = rel_rad;

      // ============================================
      // RESETAR TIMEOUT to force replan after changing experience
      // ============================================
      goal_timeout_s = 0;
      
      RCLCPP_INFO(rclcpp::get_logger("ExperienceMap"), 
                  "🔁 Pose corrected to experience %d: x=%.3f y=%.3f th=%.3f", 
                  exp_id, odom_x_, odom_y_, odom_th_);
      
      return true;
    }

    void resetGoalTimeout() {
        goal_timeout_s = 0;
    }

  template<typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & EXP_LOOPS;
      ar & EXP_CORRECTION;
      ar & MAX_GOALS;
      ar & EXP_INITIAL_EM_DEG;

      ar & experiences;
      ar & links;
      ar & goal_list;

      ar & current_exp_id & prev_exp_id;

      ar & accum_delta_facing;
      ar & accum_delta_x;
      ar & accum_delta_y;
      ar & accum_delta_time_s;

      ar & waypoint_exp_id;
      ar & goal_success;
      ar & goal_timeout_s;
      ar & goal_path_final_exp_id;
  	  
      ar & relative_rad;

    }

    double getRelativeRad() const { return relative_rad; }
    
    void setRelativeRad(double rad) { relative_rad = clip_rad_180(rad); }

private:
  friend class boost::serialization::access;

  ExperienceMap()
  {
    ;
  }
  // calculate distance between two experiences using djikstras algorithm
  // can be very slow for many experiences
  double dijkstra_distance_between_experiences(int id1, int id2);


  int EXP_LOOPS;
  double EXP_CORRECTION;
  unsigned int MAX_GOALS;
  double EXP_INITIAL_EM_DEG;

  std::vector<Experience> experiences;
  std::vector<Link> links;
  std::deque<int> goal_list;

  int current_exp_id, prev_exp_id;

  double accum_delta_facing;
  double accum_delta_x;
  double accum_delta_y;
  double accum_delta_time_s;

  double relative_rad;

  int waypoint_exp_id;
  bool goal_success;
  double goal_timeout_s;
  unsigned int goal_path_final_exp_id;

  // Pose da odometria (usada em NAVIGATION) - integrada a partir dos twists
  double odom_x_ = 0.0;
  double odom_y_ = 0.0;
  double odom_th_ = 0.0;
  bool use_odom_pose_ = false;

};

#endif // _EXPERIENCE_MAP_H_
