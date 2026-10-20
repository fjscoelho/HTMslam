from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('htmslam')

    config_file = os.path.join(pkg_dir, 'config', 'config_htmslam_spot.yaml')
    media_path = os.path.join(pkg_dir, 'media')
    image_file = 'irat_sm.tga'

    def as_bool(text):
        return text.strip().lower() in ('true', '1', 'yes', 'on')

    def launch_setup(context):
        topic_root = LaunchConfiguration('topic_root').perform(context)
        use_sim_time = as_bool(LaunchConfiguration('use_sim_time').perform(context))

        # Empty means: follow the `enable` value written in the config file.
        graphics = LaunchConfiguration('enable_graphics').perform(context).strip()
        graphics_override = {'enable': as_bool(graphics)} if graphics else {}

        # Place encoder + HTM temporal memory: PointCloud2 -> PlaceContext
        pcd_htm_node = Node(
            package='htmslam',
            executable='pcd_htm_node',
            name='pcd_htm_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'topic_root': topic_root,
                    'use_sim_time': use_sim_time
                }
            ]
        )

        # Place context cells: PlaceContext -> PlaceContextMatch (id + relative yaw)
        place_context_cells_node = Node(
            package='htmslam',
            executable='place_context_cells_node',
            name='place_context_cells_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'topic_root': topic_root,
                    'use_sim_time': use_sim_time
                }
            ]
        )

        # Pose cells: path integration corrected by the place context matches
        pose_cells_node = Node(
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
                    'use_sim_time': use_sim_time
                },
                graphics_override
            ]
        )

        # Experience map: topological map built from the pose cell actions
        experience_map_node = Node(
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
                    'use_sim_time': use_sim_time
                },
                graphics_override
            ]
        )

        return [
            pcd_htm_node,
            place_context_cells_node,
            pose_cells_node,
            experience_map_node,
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'topic_root', default_value='spot',
            description='Topic namespace shared by the pipeline nodes'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the bag/clock time (set true when playing a rosbag)'),
        DeclareLaunchArgument(
            'enable_graphics', default_value='',
            description='Leave empty to use the enable value from the config file; '
                        'true/false overrides it'),
        OpaqueFunction(function=launch_setup),
    ])
