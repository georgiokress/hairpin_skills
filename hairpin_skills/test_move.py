#!/usr/bin/env python3
"""
Erste Test-Bewegung: Roboter faehrt in die aktuelle Home-Pose
und dann 10 Grad weiter am shoulder_pan_joint.
Ziel: kompletten Stack einmal durchspielen.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class TestMove(Node):
    def __init__(self):
        super().__init__('test_move')
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.get_logger().info('Warte auf Action Server...')
        self.action_client.wait_for_server()
        self.get_logger().info('Action Server bereit.')

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

        future = self.action_client.send_goal_async(goal)
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
    node = TestMove()

    # Aktuelle Position als Home-Pose
    home = [
        0.128,   # elbow_joint
        -0.700,  # shoulder_lift_joint
        0.699,   # shoulder_pan_joint
        -1.038,  # wrist_1_joint
        -1.531,  # wrist_2_joint
        -0.356   # wrist_3_joint
    ]

    # Zielposen: shoulder_pan_joint um 10 Grad (0.175 rad) bewegen
    target = home.copy()
    target[2] = home[2] + 0.175  # shoulder_pan_joint

    input('\nEnter druecken um zu starten (Not-Aus bereithalten)...')

    node.get_logger().info('=== Fahre zu Home-Pose ===')
    node.move_to_joints(home, duration_sec=3.0)

    input('\nEnter druecken fuer naechste Bewegung...')

    node.get_logger().info('=== Fahre 10 Grad weiter ===')
    node.move_to_joints(target, duration_sec=3.0)

    input('\nEnter druecken um zurueck zu fahren...')

    node.get_logger().info('=== Zurueck zu Home-Pose ===')
    node.move_to_joints(home, duration_sec=3.0)

    node.get_logger().info('Test abgeschlossen.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()