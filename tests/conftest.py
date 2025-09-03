"""
Shared pytest fixtures for Isaac-ManipulaRL testing.

This module provides common fixtures and test utilities that can be used
across all test modules in the project.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import numpy as np


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Returns:
        pathlib.Path: Path to temporary directory that will be cleaned up after test
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """
    Provide a sample configuration dictionary for testing.
    
    Returns:
        dict: Sample configuration with common RL training parameters
    """
    return {
        "task": "BaxterCabinet",
        "num_envs": 1024,
        "sim_device": "cuda:0",
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": True,
        "multi_gpu": False,
        "seed": 42,
        "max_iterations": 1000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256
    }


@pytest.fixture
def mock_isaac_gym():
    """
    Mock Isaac Gym environment for testing without requiring actual gym installation.
    
    Returns:
        Mock: Mocked Isaac Gym with basic functionality
    """
    mock_gym = Mock()
    mock_gym.create_sim.return_value = Mock()
    mock_gym.get_sim_params.return_value = Mock()
    mock_gym.simulate.return_value = None
    mock_gym.fetch_results.return_value = None
    mock_gym.step_graphics.return_value = None
    mock_gym.draw_viewer.return_value = None
    return mock_gym


@pytest.fixture
def mock_torch_device():
    """
    Mock torch device for testing without requiring CUDA.
    
    Returns:
        Mock: Mocked torch device
    """
    mock_device = Mock()
    mock_device.__str__ = Mock(return_value="cuda:0")
    return mock_device


@pytest.fixture
def sample_observation():
    """
    Generate sample observation data for RL testing.
    
    Returns:
        numpy.ndarray: Sample observation array
    """
    return np.random.random((10, 64)).astype(np.float32)


@pytest.fixture
def sample_action():
    """
    Generate sample action data for RL testing.
    
    Returns:
        numpy.ndarray: Sample action array
    """
    return np.random.random((10, 8)).astype(np.float32)


@pytest.fixture
def mock_neural_network():
    """
    Mock neural network for testing RL algorithms.
    
    Returns:
        Mock: Mocked neural network with forward pass
    """
    mock_net = Mock()
    mock_net.forward.return_value = (Mock(), Mock(), Mock())
    mock_net.parameters.return_value = []
    mock_net.state_dict.return_value = {}
    mock_net.load_state_dict.return_value = None
    return mock_net


@pytest.fixture
def mock_environment():
    """
    Mock RL environment for testing algorithms without full simulation.
    
    Returns:
        Mock: Mocked environment with step, reset, and observation methods
    """
    mock_env = Mock()
    mock_env.num_envs = 1024
    mock_env.num_obs = 64
    mock_env.num_acts = 8
    mock_env.reset.return_value = np.random.random((1024, 64)).astype(np.float32)
    mock_env.step.return_value = (
        np.random.random((1024, 64)).astype(np.float32),  # observations
        np.random.random(1024).astype(np.float32),        # rewards
        np.random.choice([True, False], 1024),            # dones
        {}                                                # info
    )
    return mock_env


@pytest.fixture
def robot_joint_names():
    """
    Provide sample robot joint names for testing.
    
    Returns:
        list: List of joint names for common robots
    """
    return {
        "baxter": [
            "left_s0", "left_s1", "left_e0", "left_e1", 
            "left_w0", "left_w1", "left_w2",
            "right_s0", "right_s1", "right_e0", "right_e1", 
            "right_w0", "right_w1", "right_w2"
        ],
        "ur5": [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
    }


@pytest.fixture
def sample_trajectory():
    """
    Generate sample trajectory data for testing motion planning.
    
    Returns:
        dict: Sample trajectory with positions, velocities, and timestamps
    """
    time_steps = 100
    num_joints = 7
    
    return {
        "positions": np.random.random((time_steps, num_joints)).astype(np.float32),
        "velocities": np.random.random((time_steps, num_joints)).astype(np.float32),
        "accelerations": np.random.random((time_steps, num_joints)).astype(np.float32),
        "timestamps": np.linspace(0, 10, time_steps).astype(np.float32)
    }


@pytest.fixture
def mock_ros_node():
    """
    Mock ROS node functionality for testing ROS integration.
    
    Returns:
        Mock: Mocked ROS node with common methods
    """
    mock_node = Mock()
    mock_node.get_param.return_value = "test_param"
    mock_node.set_param.return_value = None
    mock_node.publish.return_value = None
    mock_node.subscribe.return_value = None
    mock_node.spin.return_value = None
    return mock_node


@pytest.fixture
def sample_point_cloud():
    """
    Generate sample point cloud data for testing perception components.
    
    Returns:
        numpy.ndarray: Sample 3D point cloud data
    """
    num_points = 1000
    return np.random.random((num_points, 3)).astype(np.float32)


@pytest.fixture
def mock_camera_data():
    """
    Mock camera data for testing vision components.
    
    Returns:
        dict: Mocked camera data with RGB and depth images
    """
    return {
        "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "depth": np.random.random((480, 640)).astype(np.float32),
        "camera_matrix": np.eye(3, dtype=np.float32),
        "distortion_coeffs": np.zeros(5, dtype=np.float32)
    }


@pytest.fixture
def cleanup_files():
    """
    Cleanup fixture to remove test files after test execution.
    
    Yields:
        list: List to add file paths that should be cleaned up
    """
    files_to_cleanup = []
    yield files_to_cleanup
    
    for file_path in files_to_cleanup:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (takes more than a few seconds)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)