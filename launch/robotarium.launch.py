from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
from ament_index_python.packages import get_package_share_directory

# ros2 bag play _2022-04-07-14-14-35_robotarium/_2022-04-07-14-14-35_robotarium_ros2.db3 --clock --start-paused --remap /stereo_camera/left/image_raw:=/robotarium/camera/image /odometry/filtered:=/robotarium/odom

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('htmslam')
    
    # Configuration file
    config_file = os.path.join(pkg_dir, 'config', 'config_htmslam_robotarium.yaml')
    
    # Common parameters
    topic_root = 'robotarium'
    media_path = os.path.join(pkg_dir, 'media')
    image_file = 'irat_sm.tga'
    
    # Python module path for visual feature extractor
    python_module_path = os.path.join(pkg_dir, '..', '..', 'lib', 'htmslam', 'visual_feature_extractor')
    
    # Random matrix path for binary projector
    random_matrix_path = os.path.join(pkg_dir, 'random_matrix', 'randomMatrix.bin')
    
    # NeoSLAM nodes

    # New: Add mode manager
    mode_manager_node = Node(
        package='htmslam',
        executable='mode_manager_node',  # Nome do executável
        name='mode_manager',
        output='screen',
        parameters=[{
            'initial_mode': 'mapping'  # Modo inicial
        }]
    )

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

    # ============================================
    # New: Publish static TF map -> odom
    # ============================================
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    return LaunchDescription([
        mode_manager_node,  # <-- ADICIONADO
        vfe_node,
        bp_node,
        neocortex_node,
        svc_node,
        pc_node,
        em_node,
        static_tf_node  # <-- ADICIONADO
    ])