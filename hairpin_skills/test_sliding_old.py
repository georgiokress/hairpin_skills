#!/usr/bin/env python3
"""
Test S3 Touchdown + S4 Sliding:
Approach -> Touchdown -> Sliding Richtung Nut -> Home
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import WrenchStamped
from ur_msgs.srv import SetForceMode, SetIO
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger
import time


class TestSliding(Node):
    def __init__(self):
        super().__init__('test_sliding')

        self.move_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.io_client = self.create_client(
            SetIO, '/io_and_status_controller/set_io')
        self.force_mode_client = self.create_client(
            SetForceMode, '/force_mode_controller/start_force_mode')
        self.force_stop_client = self.create_client(
            Trigger, '/force_mode_controller/stop_force_mode')
        self.switch_ctrl_client = self.create_client(
            SwitchController, '/controller_manager/switch_controller')

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
            msg.wrench.force.x, msg.wrench.force.y,
            msg.wrench.force.z, msg.wrench.torque.x,
            msg.wrench.torque.y, msg.wrench.torque.z]

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
        req.strictness = 2
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
            'elbow_joint', 'shoulder_lift_joint',
            'shoulder_pan_joint', 'wrist_1_joint',
            'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec), nanosec=0)
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

    # === Force Mode: Touchdown (nur Z) ===
    def start_force_mode_z(self, force_z, speed_limit=0.01):
        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base'
        req.task_frame.pose.orientation.w = 1.0
        req.selection_vector_z = True
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
                f'Force-Mode Z gestartet: F_z={force_z}N')
        else:
            self.get_logger().error('Force-Mode Z fehlgeschlagen!')
        return result.success

    # === Force Mode: Sliding (Z + Y Richtung Mittelpunkt) ===
    def start_force_mode_sliding(self, center_x, center_y, center_z,
                                  force_z=-10.0, force_y=-5.0,
                                  speed_limit=0.01):
        # Slide-Richtung berechnen: vom TCP zum Mittelpunkt im Base-Frame
        # Aktuelle TCP-Position aus Wrench-Callback nicht verfuegbar,
        # daher nutzen wir die Approach-Pose als Naeherung
        import math
        tcp_x = -0.2356  # Approach-Pose x
        tcp_y = -0.4109  # Approach-Pose y

        dx = center_x - tcp_x
        dy = center_y - tcp_y
        length = math.sqrt(dx**2 + dy**2)
        dir_x = dx / length
        dir_y = dy / length

        # Kraft in Slide-Richtung aufteilen auf X und Y im Base-Frame
        slide_force = abs(force_y)
        fx = dir_x * slide_force
        fy = dir_y * slide_force

        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base'
        req.task_frame.pose.orientation.w = 1.0
        req.selection_vector_x = True
        req.selection_vector_y = True
        req.selection_vector_z = True
        req.wrench.force.x = fx
        req.wrench.force.y = fy
        req.wrench.force.z = force_z   # nach unten im Base-Frame
        req.type = 2  # NO_TRANSFORM — Base-Frame direkt
        req.speed_limits.linear.x = speed_limit
        req.speed_limits.linear.y = speed_limit
        req.speed_limits.linear.z = speed_limit
        req.speed_limits.angular.x = 0.005
        req.speed_limits.angular.y = 0.005
        req.speed_limits.angular.z = 0.005
        req.deviation_limits = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        req.damping_factor = 0.5
        req.gain_scaling = 0.5

        future = self.force_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.success:
            self.get_logger().info(
                f'Sliding gestartet (type=2): Fx={fx:.2f}N, '
                f'Fy={fy:.2f}N, Fz={force_z:.2f}N')
        else:
            self.get_logger().error('Sliding Force-Mode fehlgeschlagen!')
        return result.success

    def stop_force_mode(self):
        future = self.force_stop_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Force-Mode gestoppt.')

    # === S3 Touchdown ===
    def touchdown(self, force_z=-10.0, f_threshold=6.0, timeout=10.0):
        self.get_logger().info(
            f'=== S3 Touchdown: F_z={force_z}N, '
            f'Threshold={f_threshold}N ===')

        if not self.start_force_mode_z(force_z, speed_limit=0.01):
            return False

        time.sleep(0.5)
        start_time = time.time()
        contact = False

        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            fz = abs(self.current_wrench[2])
            if fz > f_threshold:
                self.get_logger().info(
                    f'KONTAKT! F_z={self.current_wrench[2]:.2f}N')
                contact = True
                break

        self.stop_force_mode()
        if not contact:
            self.get_logger().warn(f'Timeout nach {timeout}s!')
        return contact

    # === S4 Sliding ===
    def sliding(self, center_x, center_y, center_z,
                force_z=-10.0, force_y=-5.0,
                f_xy_threshold=8.0, timeout=10.0):
        self.get_logger().info(
            f'=== S4 Sliding: F_z={force_z}N, F_y={force_y}N, '
            f'Threshold F_xy={f_xy_threshold}N ===')

        if not self.start_force_mode_sliding(
                center_x, center_y, center_z,
                force_z, force_y, speed_limit=0.01):
            return False

        time.sleep(0.5)
        start_time = time.time()
        slot_found = False

       # Baseline F_z messen (auf der Oberflaeche)
        rclpy.spin_once(self, timeout_sec=0.1)
        fz_baseline = abs(self.current_wrench[2])
        self.get_logger().info(f'F_z Baseline: {fz_baseline:.2f}N')

        start_time = time.time()
        slot_found = False

        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            fz_current = abs(self.current_wrench[2])

            # Fase erkannt wenn F_z um mehr als 2N abfaellt
            if fz_current < fz_baseline - 2.0:
                self.get_logger().info(
                    f'FASE GEFUNDEN! F_z={fz_current:.2f}N '
                    f'(Baseline war {fz_baseline:.2f}N)')
                slot_found = True
                break

        self.stop_force_mode()
        if not slot_found:
            self.get_logger().warn(f'Timeout nach {timeout}s!')
        return slot_found
      


# def main():
#     rclpy.init()
#     node = TestSliding()

#     home = [0.128, -0.700, 0.699, -1.038, -1.531, -0.356]
#     approach_rotated = [1.299, -1.083, 0.915, -1.971, -1.571, 0.666]

#     # Netzmittelpunkt
#     center_x = -0.3109
#     center_y = -0.3556
#     center_z = 0.1424

#     # Schritt 1: Pin einlegen
#     input('\nPin manuell einlegen, dann Enter...')
#     node.gripper_close()
#     time.sleep(1.0)

#     # Schritt 2: Approach-Pose
#     input('\nEnter fuer Approach-Pose (Not-Aus bereithalten)...')
#     node.move_to_joints(approach_rotated, duration_sec=4.0)

#     # Schritt 3: Touchdown
#     input('\nEnter fuer Touchdown...')
#     node.switch_to_force_mode()
#     contact = node.touchdown(force_z=-10.0, f_threshold=6.0, timeout=10.0)

#     if not contact:
#         node.get_logger().error('Kein Kontakt - Abbruch!')
#         node.switch_to_trajectory()
#         node.move_to_joints(home, duration_sec=4.0)
#         #node.gripper_open()
#         node.destroy_node()
#         rclpy.shutdown()
#         return

#     # Schritt 4: Sliding Richtung Nut
#     input('\nEnter fuer Sliding Richtung Nut...')
#     slot_found = node.sliding(
#         center_x, center_y, center_z,
#         force_z=-5.0, force_y=-10.0,
#         timeout=10.0)

#     if slot_found:
#         node.get_logger().info('Nut gefunden!')
#     else:
#         node.get_logger().warn('Nut nicht gefunden.')

#     # Schritt 5: Zurueck zu Home
#     input('\nEnter fuer zurueck zu Home...')
#     node.switch_to_trajectory()
#     node.move_to_joints(home, duration_sec=4.0)

#     # Schritt 6: Greifer oeffnen
#     #input('\nEnter fuer Greifer oeffnen...')
#     #node.gripper_open()

#     node.get_logger().info('Test abgeschlossen.')
#     node.destroy_node()
#     rclpy.shutdown()

def main():
    rclpy.init()
    node = TestSliding()

    home = [0.128, -0.700, 0.699, -1.038, -1.531, -0.356]
    approach_rotated = [1.299, -1.083, 0.915, -1.971, -1.571, 0.666]

    # Netzmittelpunkt
    center_x = -0.3109
    center_y = -0.3556
    center_z = 0.1424

    # Schritt 1: Pin einlegen
    input('\nPin manuell einlegen, dann Enter...')
    node.gripper_close()
    time.sleep(1.0)

    # Schritt 2: Approach-Pose
    input('\nEnter fuer Approach-Pose (Not-Aus bereithalten)...')
    if not node.move_to_joints(approach_rotated, duration_sec=4.0):
        node.get_logger().error('Approach fehlgeschlagen - Abbruch!')
        node.destroy_node()
        rclpy.shutdown()
        return

    # Schritt 3: Touchdown
    input('\nEnter fuer Touchdown...')
    node.switch_to_force_mode()
    contact = node.touchdown(force_z=-10.0, f_threshold=6.0, timeout=10.0)
    node.switch_to_trajectory()   # immer zurueck, egal ob Kontakt oder nicht
    time.sleep(0.5)

    if not contact:
        node.get_logger().error('Kein Kontakt - fahre zu Home!')
        node.move_to_joints(home, duration_sec=4.0)
        node.destroy_node()
        rclpy.shutdown()
        return

    # Schritt 4: Sliding
    input('\nEnter fuer Sliding Richtung Nut...')
    node.switch_to_force_mode()
    slot_found = node.sliding(
        center_x, center_y, center_z,
        force_z=-5.0, force_y=-10.0,
        timeout=10.0)
    node.switch_to_trajectory()   # immer zurueck
    time.sleep(0.5)

    if slot_found:
        node.get_logger().info('Nut gefunden!')
    else:
        node.get_logger().warn('Nut nicht gefunden.')

    # Schritt 5: Zurueck zu Home
    input('\nEnter fuer zurueck zu Home...')
    if not node.move_to_joints(home, duration_sec=4.0):
        node.get_logger().error('Home-Fahrt fehlgeschlagen!')

    node.get_logger().info('Test abgeschlossen.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()