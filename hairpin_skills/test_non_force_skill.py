#!/usr/bin/env python3
"""
Test: Pin manuell einlegen, Greifer schliessen,
zur rotierten Approach-Pose fahren, zurueck, Greifer oeffnen.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ur_msgs.srv import SetIO
import time


class TestApproach(Node):
    def __init__(self):
        super().__init__('test_approach')

        # Action Client fuer Bewegung
        self.move_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Service Client fuer Greifer IO
        self.io_client = self.create_client(
            SetIO,
            '/io_and_status_controller/set_io'
        )

        self.get_logger().info('Warte auf Action Server...')
        self.move_client.wait_for_server()
        self.get_logger().info('Warte auf IO Service...')
        self.io_client.wait_for_service()
        self.get_logger().info('Alles bereit.')

    def set_io(self, pin, state):
        """Setzt einen digitalen Ausgang."""
        req = SetIO.Request()
        req.fun = 1       # digitaler Ausgang
        req.pin = pin
        req.state = float(state)
        future = self.io_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        time.sleep(0.3)   # kurz warten bis Ventil geschaltet hat

    def gripper_open(self):
        self.get_logger().info('Greifer oeffnen...')
        self.set_io(1, 0)   # Schliessen LOW
        self.set_io(0, 1)   # Oeffnen HIGH

    def gripper_close(self):
        self.get_logger().info('Greifer schliessen...')
        self.set_io(0, 0)   # Oeffnen LOW
        self.set_io(1, 1)   # Schliessen HIGH

    def move_to_joints(self, positions, duration_sec=5.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'elbow_joint',
            'shoulder_lift_joint',
            'shoulder_pan_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=0
        )
        goal.trajectory.points = [point]

        self.get_logger().info(
            f'Fahre zu: {[round(p, 3) for p in positions]}')

        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal abgelehnt!')
            return False

        self.get_logger().info('Bewegung laeuft...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Bewegung abgeschlossen.')
        return True


def main():
    rclpy.init()
    node = TestApproach()

    home = [0.128, -0.700, 0.699, -1.038, -1.531, -0.356]
    approach_rotated = [1.299, -1.083, 0.915, -1.971, -1.571, 0.666]

    # Schritt 1: Greifer schliessen (Pin manuell eingelegt)
    input('\nPin manuell einlegen, dann Enter...')
    node.gripper_close()
    time.sleep(1.0)

    # Schritt 2: Zur Home-Pose fahren
    input('\nEnter fuer Home-Pose (Not-Aus bereithalten)...')
    node.move_to_joints(home, duration_sec=4.0)

    # Schritt 3: Zur rotierten Approach-Pose fahren
    input('\nEnter fuer Approach-Pose (rotiert, 20mm ueber Slot)...')
    node.move_to_joints(approach_rotated, duration_sec=4.0)

    # Schritt 4: Zurueck zur Home-Pose
    input('\nEnter fuer zurueck zu Home...')
    node.move_to_joints(home, duration_sec=4.0)

    # Schritt 5: Greifer oeffnen
    input('\nEnter fuer Greifer oeffnen...')
    node.gripper_open()

    node.get_logger().info('Test abgeschlossen.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()