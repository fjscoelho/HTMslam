from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

# ros2 bag play ros1_2026_04_09
#  --clock --start-paused --remap /D400/color/image_raw:=/ROSI/camera/image --rate 1.0

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('htmslam')
    
    # Configuration file
    config_file = os.path.join(pkg_dir, 'config', 'config_htmslam_ROSI_t265_fusedVO.yaml')
    
    # Common parameters
    topic_root = 'ROSI'
    media_path = os.path.join(pkg_dir, 'media')
    image_file = 'irat_sm.tga'
    
    # Python module path for visual feature extractor
    python_module_path = os.path.join(pkg_dir, '..', '..', 'lib', 'htmslam', 'visual_feature_extractor')
    
    # Random matrix path for binary projector
    random_matrix_path = os.path.join(pkg_dir, 'random_matrix', 'randomMatrix.bin')
    
    # NeoSLAM nodes
    vfe_node = Node(
        package='htmslam',
        executable='visual_feature_extractor_node',
        name='visual_feature_extractor_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'python_module_path': python_module_path,
                'use_sim_time': False
            }
        ]
    )

    bp_node = Node(
        package='htmslam',
        executable='binary_projector_node',
        name='binary_projector_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'random_matrix_path': random_matrix_path,
                'use_sim_time': False
            }
        ]
    )

    neocortex_node = Node(
        package='htmslam',
        executable='neocortex_node',
        name='neocortex_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'use_sim_time': False
            }
        ]
    )

    svc_node = Node(
        package='htmslam',
        executable='spatial_view_cells_node',
        name='spatial_view_cells_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'use_sim_time': False
            }
        ]
    )

    pc_node = Node(
        package='htmslam',
        executable='pose_cells_node',
        name='pose_cells_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'media_path': media_path,
                'image_file': image_file,
                'use_sim_time': False
            }
        ]
    )

    em_node = Node(
        package='htmslam',
        executable='experience_map_node',
        name='experience_map_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'media_path': media_path,
                'image_file': image_file,
                'use_sim_time': False
            }
        ]
    )

    vo_node = Node(
        package='htmslam',
        executable='visual_odometry_node',
        name='visual_odometry_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'media_path': media_path,
                'image_file': image_file,
                'use_sim_time': False
            }
        ]
    )

    st_model_node = Node(
        package='htmslam',
        executable='steering_model_node',
        name='steering_model_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'media_path': media_path,
                'image_file': image_file,
                'use_sim_time': False
            }
        ]
    )
    odom_fusion_node = Node(
        package='htmslam',
        executable='odom_fusion_node',
        name='odom_fusion_node',
        output='screen',
        parameters=[
            config_file,
            {
                'topic_root': topic_root,
                'media_path': media_path,
                'image_file': image_file,
                'use_sim_time': False
            }
        ]
    )

    return LaunchDescription([
        vfe_node,
        bp_node,
        neocortex_node,
        svc_node,
        pc_node,
        em_node,
        vo_node,
        st_model_node,
        odom_fusion_node
    ])
