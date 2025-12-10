#!/usr/bin/env python3
"""
Quick test script to verify Gazebo can launch with TurtleBot3.
"""

import sys
import time

# Add the pykal module to the path
sys.path.insert(0, '/home/nehal/.emacs.d/Projects/Software/pykal/src')

from pykal.utilities.gazebo import start_gazebo, stop_gazebo

def test_gazebo_launch():
    """Test that Gazebo can launch successfully."""
    print("=" * 60)
    print("Testing TurtleBot3 Gazebo Launch")
    print("=" * 60)

    try:
        # Launch Gazebo in headless mode for faster testing
        print("\n[1/3] Launching Gazebo...")
        gz = start_gazebo(
            robot='turtlebot3',
            world='turtlebot3_world',
            headless=True,  # No GUI for testing
            model='burger',
            verbose=True
        )

        print("\n[2/3] Gazebo launched! Process info:")
        print(f"  {gz}")
        print(f"  Running: {gz.is_running()}")

        # Keep it running for a few seconds
        print("\n[3/3] Keeping Gazebo running for 10 seconds...")
        time.sleep(10)

        # Stop Gazebo
        print("\nStopping Gazebo...")
        stop_gazebo(gz, verbose=True)

        print("\n" + "=" * 60)
        print("✓ TEST PASSED: Gazebo launched and stopped successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        return False

if __name__ == '__main__':
    success = test_gazebo_launch()
    sys.exit(0 if success else 1)
