#!/bin/bash

# Dependency installation script for htmslam
# ROS 2 Jazzy on Ubuntu

echo "======================================"
echo "htmslam ROS 2 - Dependency Installation"
echo "======================================"
echo ""

# Check if ROS 2 is installed
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "✓ ROS 2 Jazzy found"
    source /opt/ros/jazzy/setup.bash
else
    echo "✗ ROS 2 Jazzy not found!"
    echo "Please install ROS 2 Jazzy first:"
    echo "https://docs.ros.org/en/jazzy/Installation.html"
    exit 1
fi

echo ""
echo "Installing system dependencies..."
echo ""

# Update package list
sudo apt update

# Install ROS 2 dependencies
echo "Installing ROS 2 packages..."
sudo apt install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-vision-opencv

# Install OpenCV
echo "Installing OpenCV..."
sudo apt install -y \
    libopencv-dev \
    python3-opencv

# Install Boost
echo "Installing Boost..."
sudo apt install -y \
    libboost-all-dev

# Install Irrlicht (optional, for visualization)
echo "Installing Irrlicht..."
sudo apt install -y \
    libirrlicht-dev

# Install OpenGL
echo "Installing OpenGL..."
sudo apt install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev

# Install build tools
echo "Installing build tools..."
sudo apt install -y \
    build-essential \
    cmake

echo ""
echo "======================================"
echo "✓ Installation completed!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy the package to your workspace:"
echo "   mkdir -p ~/ros2_ws/src"
echo "   cp -r /path/to/htmslam ~/ros2_ws/src/"
echo ""
echo "2. Build the package:"
echo "   cd ~/ros2_ws"
echo "   colcon build --packages-select htmslam"
echo ""
echo "3. Source the workspace:"
echo "   source install/setup.bash"
echo ""
echo "4. Run:"
echo "   ros2 launch htmslam stlucia.launch.py"
echo ""
