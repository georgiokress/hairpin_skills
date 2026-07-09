#!/usr/bin/env python3
"""
Test S3 Touchdown: Force-Mode nach unten bis Kontakt.
Sequenz: Home -> Approach (Joint) -> Controller-Wechsel ->
         Force-Mode Z nach unten -> Stop bei F_threshold
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import WrenchStamped, PoseStamped, Wrench, Twist
from ur_msgs.srv import SetForceMode, SetIO
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger
import time


class TestTouchdown(Node):
    def __init__(self):
        super().__init__('test_touchdown')

        # Action Client fuer Joint-Bewegung
        self.move_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Service Clients
        self.io_client = self.create_client(
            SetIO, '/io_and_status_controller/set_io')
        self.force_mode_client = self.create_client(
            SetForceMode, '/force_mode_controller/start_force_mode')
        self.force_stop_client = self.create_client(
            Trigger, '/force_mode_controller/stop_force_mode')
        self.switch_ctrl_client = self.create_client(
            SwitchController, '/controller_manager/switch_controller')

        # Wrench Subscriber
        self.current_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self._wrench_cb, 10)

        self.get_logger().info('Warte auf Services...')
        self.move_client.wait_for_server()
        self.io_client.wait_for_service()
        self.force_mode_client.wait_for_service()
        self.force_stop_client.wait_for_service()
        self.switch_ctrl_client.wait_for_service()
        self.get_logger().info('Alles bereit.')

    def _wrench_cb(self, msg):
        self.current_wrench = [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ]

    # === IO / Greifer ===
    def set_io(self, pin, state):
        req = SetIO.Request()
        req.fun = 1
        req.pin = pin
        req.state = float(state)
        future = self.io_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        time.sleep(0.3)

    def gripper_close(self):
        self.get_logger().info('Greifer schliessen...')
        self.set_io(0, 0)
        self.set_io(1, 1)

    def gripper_open(self):
        self.get_logger().info('Greifer oeffnen...')
        self.set_io(1, 0)
        self.set_io(0, 1)

    # === Controller Wechsel ===
    def switch_to_force_mode(self):
        self.get_logger().info('Wechsle zu Force-Mode-Controller...')
        req = SwitchController.Request()
        req.activate_controllers = ['force_mode_controller']
        req.deactivate_controllers = ['scaled_joint_trajectory_controller']
        req.strictness = 2  # STRICT
        future = self.switch_ctrl_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.ok:
            self.get_logger().info('Force-Mode-Controller aktiv.')
        else:
            self.get_logger().error('Controller-Wechsel fehlgeschlagen!')
        time.sleep(0.5)
        return result.ok

    def switch_to_trajectory(self):
        self.get_logger().info('Wechsle zu Trajectory-Controller...')
        req = SwitchController.Request()
        req.activate_controllers = ['scaled_joint_trajectory_controller']
        req.deactivate_controllers = ['force_mode_controller']
        req.strictness = 2
        future = self.switch_ctrl_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.ok:
            self.get_logger().info('Trajectory-Controller aktiv.')
        else:
            self.get_logger().error('Controller-Wechsel fehlgeschlagen!')
        time.sleep(0.5)
        return result.ok

    # === Joint-Bewegung ===
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
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=0)
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

    # === Force Mode ===
    def start_force_mode_z(self, force_z, speed_limit=0.01):
        """Startet Force-Mode nur in Z-Richtung."""
        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base'
        req.task_frame.pose.orientation.w = 1.0
        req.selection_vector_x = False
        req.selection_vector_y = False
        req.selection_vector_z = True
        req.selection_vector_rx = False
        req.selection_vector_ry = False
        req.selection_vector_rz = False
        req.wrench.force.z = force_z
        req.type = 2
        req.speed_limits.linear.x = speed_limit
        req.speed_limits.linear.y = speed_limit
        req.speed_limits.linear.z = speed_limit
        req.speed_limits.angular.x = 0.5
        req.speed_limits.angular.y = 0.5
        req.speed_limits.angular.z = 0.5
        req.deviation_limits = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        req.damping_factor = 0.025
        req.gain_scaling = 0.5

        future = self.force_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.success:
            self.get_logger().info(
                f'Force-Mode gestartet: F_z={force_z} N')
        else:
            self.get_logger().error('Force-Mode Start fehlgeschlagen!')
        return result.success

    def stop_force_mode(self):
        future = self.force_stop_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Force-Mode gestoppt.')

    # === Touchdown ===
    def touchdown(self, force_z=-10.0, f_threshold=5.0, timeout=10.0):
        """
        S3 Touchdown: Force-Mode nach unten bis |F_z| > threshold.
        """
        self.get_logger().info(
            f'=== S3 Touchdown: F_z={force_z}N, '
            f'Threshold={f_threshold}N ===')

        if not self.start_force_mode_z(force_z, speed_limit=0.01):
            return False

        # Kurz warten bis Wrench-Daten aktuell sind
        time.sleep(0.5)

        # Kraft ueberwachen
        start_time = time.time()
        contact = False

        while time.time() - start_time < timeout:
            # Wrench Callback muss feuern
            rclpy.spin_once(self, timeout_sec=0.01)

            fz = abs(self.current_wrench[2])

            if fz > f_threshold:
                self.get_logger().info(
                    f'KONTAKT! F_z={self.current_wrench[2]:.2f} N '
                    f'(Threshold={f_threshold} N)')
                contact = True
                break

        self.stop_force_mode()

        if not contact:
            self.get_logger().warn(
                f'Timeout nach {timeout}s ohne Kontakt!')

        return contact


def main():
    rclpy.init()
    node = TestTouchdown()

    home = [0.128, -0.700, 0.699, -1.038, -1.531, -0.356]
    approach_rotated = [1.299, -1.083, 0.915, -1.971, -1.571, 0.666]

    # Schritt 1: Pin einlegen und Greifer schliessen
    input('\nPin manuell einlegen, dann Enter...')
    node.gripper_close()
    time.sleep(1.0)

    # Schritt 2: Zur Approach-Pose fahren (Joint-Controller)
    input('\nEnter fuer Approach-Pose (Not-Aus bereithalten)...')
    node.move_to_joints(approach_rotated, duration_sec=4.0)

    # Schritt 3: Controller wechseln und Touchdown
    input('\nEnter fuer Touchdown (Roboter faehrt langsam nach unten)...')
    node.switch_to_force_mode()
    contact = node.touchdown(force_z=-10.0, f_threshold=5.0, timeout=10.0)

    if contact:
        node.get_logger().info('Touchdown erfolgreich!')
    else:
        node.get_logger().warn('Kein Kontakt erkannt.')

    # Schritt 4: Zurueck - Controller wechseln und Home
    input('\nEnter fuer zurueck zu Home...')
    node.switch_to_trajectory()
    node.move_to_joints(home, duration_sec=4.0)

    # Schritt 5: Greifer oeffnen
    input('\nEnter fuer Greifer oeffnen...')
    node.gripper_open()

    node.get_logger().info('Test abgeschlossen.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()