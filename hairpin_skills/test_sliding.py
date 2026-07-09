#!/usr/bin/env python3
"""
Test S3 Touchdown + S4 Sliding mit Stabilisierungsphase:
Approach -> Touchdown -> Stabilisierung (-3N) -> Sliding -> Home

Termination-Strategie:
- Feste Anpresskraft -3N als Baseline
- 250ms Stabilisierung vor Sliding-Start
- Tiefpass-Filter (gleitender Mittelwert ueber 5 Samples)
- Mindestdauer 30ms unter Threshold zur Auslösung
- Mindestlaufzeit 500ms bevor Termination scharf wird
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
from collections import deque
import math
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

    # === Force Mode: Sliding (X+Y+Z im Base-Frame) ===
    def start_force_mode_sliding(self, center_x, center_y, center_z,
                                  force_z=-3.0, slide_force=0.0,
                                  speed_limit=0.01):
        """
        Force-Mode fuer Sliding-Phase.
        slide_force = 0  -> nur Anpresskraft (Stabilisierungsphase)
        slide_force > 0  -> Anpresskraft + laterale Kraft (Sliding)
        """
        # Slide-Richtung berechnen: vom Approach-Punkt zum Mittelpunkt
        tcp_x = -0.2356  # Approach-Pose x
        tcp_y = -0.4109  # Approach-Pose y

        dx = center_x - tcp_x
        dy = center_y - tcp_y
        length = math.sqrt(dx**2 + dy**2)
        dir_x = dx / length
        dir_y = dy / length

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
        req.wrench.force.z = force_z
        req.type = 2  # NO_TRANSFORM - Base-Frame direkt
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
                f'Force-Mode gestartet: Fx={fx:.2f}N, '
                f'Fy={fy:.2f}N, Fz={force_z:.2f}N')
        else:
            self.get_logger().error('Force-Mode fehlgeschlagen!')
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

    # === S4 Sliding mit Stabilisierungsphase ===
    def sliding(self, center_x, center_y, center_z,
                anpresskraft=-3.0, slide_force=10.0,
                stabilisierung_ms=250, baseline_ms=50,
                drop_ratio=0.4, min_under_ms=30,
                ramp_up_ms=500, timeout=10.0):
        """
        Sliding-Skill mit drei Phasen:
        A) Stabilisierung: nur Anpresskraft -3N, 250ms einschwingen lassen
        B) Baseline-Messung: F_z 50ms mitteln
        C) Sliding: laterale Kraft + Termination ueber F_z-Drop

        Parameter:
            anpresskraft:      Konstanter F_z-Sollwert (z.B. -3 N)
            slide_force:       Betrag der lateralen Kraft Richtung Mitte
            stabilisierung_ms: Dauer Phase A
            baseline_ms:       Dauer Phase B (F_z mitteln)
            drop_ratio:        Termination wenn |F_z| < baseline * drop_ratio
                               (0.4 = 60% Drop)
            min_under_ms:      Mindestdauer unter Threshold fuer Auslösung
            ramp_up_ms:        Mindestlaufzeit bevor Termination scharf wird
            timeout:           Gesamt-Timeout fuer Phase C
        """
        self.get_logger().info(
            f'=== S4 Sliding: Anpresskraft={anpresskraft}N, '
            f'Slide={slide_force}N ===')

        # === Phase A: Stabilisierung ===
        self.get_logger().info(
            f'Phase A: Stabilisierung ({stabilisierung_ms}ms) mit '
            f'F_z={anpresskraft}N')
        if not self.start_force_mode_sliding(
                center_x, center_y, center_z,
                force_z=anpresskraft, slide_force=0.0,
                speed_limit=0.01):
            return False

        stab_start = time.time()
        while (time.time() - stab_start) * 1000 < stabilisierung_ms:
            rclpy.spin_once(self, timeout_sec=0.01)

        # === Phase B: Baseline messen ===
        self.get_logger().info(
            f'Phase B: Baseline-Messung ({baseline_ms}ms)')
        baseline_samples = []
        baseline_start = time.time()
        while (time.time() - baseline_start) * 1000 < baseline_ms:
            rclpy.spin_once(self, timeout_sec=0.005)
            baseline_samples.append(abs(self.current_wrench[2]))

        if not baseline_samples:
            self.get_logger().error('Keine Baseline-Samples!')
            self.stop_force_mode()
            return False

        fz_baseline = sum(baseline_samples) / len(baseline_samples)
        threshold = fz_baseline * drop_ratio
        self.get_logger().info(
            f'Baseline: {fz_baseline:.2f}N, '
            f'Drop-Threshold: {threshold:.2f}N')

        # === Phase C: Sliding mit Termination ===
        self.get_logger().info(
            f'Phase C: Sliding mit Slide-Kraft {slide_force}N')
        # Neuer Force-Mode mit lateraler Kraft
        if not self.start_force_mode_sliding(
                center_x, center_y, center_z,
                force_z=anpresskraft, slide_force=slide_force,
                speed_limit=0.01):
            return False

        # Tiefpass-Filter: gleitender Mittelwert ueber 5 Samples
        fz_window = deque(maxlen=5)
        under_threshold_start = None
        slot_found = False
        slide_start = time.time()

        while time.time() - slide_start < timeout:
            rclpy.spin_once(self, timeout_sec=0.005)
            fz_current = abs(self.current_wrench[2])
            fz_window.append(fz_current)

            if len(fz_window) < fz_window.maxlen:
                continue

            fz_filtered = sum(fz_window) / len(fz_window)
            elapsed_ms = (time.time() - slide_start) * 1000

            # Termination erst nach ramp_up_ms scharf
            if elapsed_ms < ramp_up_ms:
                continue

            # Pruefen ob unter Threshold
            if fz_filtered < threshold:
                if under_threshold_start is None:
                    under_threshold_start = time.time()
                elif (time.time() - under_threshold_start) * 1000 >= min_under_ms:
                    self.get_logger().info(
                        f'FASE GEFUNDEN! Gefilterte F_z={fz_filtered:.2f}N '
                        f'(Baseline {fz_baseline:.2f}N, '
                        f'Threshold {threshold:.2f}N)')
                    slot_found = True
                    break
            else:
                under_threshold_start = None  # Reset wenn wieder ueber

        self.stop_force_mode()
        if not slot_found:
            self.get_logger().warn(
                f'Timeout nach {timeout}s ohne Fase!')
        return slot_found


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
    node.switch_to_trajectory()
    time.sleep(0.5)

    if not contact:
        node.get_logger().error('Kein Kontakt - fahre zu Home!')
        node.move_to_joints(home, duration_sec=4.0)
        node.destroy_node()
        rclpy.shutdown()
        return

    # Schritt 4: Sliding mit Stabilisierungsphase
    input('\nEnter fuer Sliding Richtung Nut...')
    node.switch_to_force_mode()
    slot_found = node.sliding(
        center_x, center_y, center_z,
        anpresskraft=-3.0,       # konstante Anpresskraft
        slide_force=10.0,         # laterale Schubkraft
        stabilisierung_ms=250,    # Einschwingphase
        baseline_ms=50,           # Baseline-Messung
        drop_ratio=0.4,           # 60% Drop = Fase
        min_under_ms=30,          # 30ms unter Threshold
        ramp_up_ms=500,           # Termination erst nach 500ms scharf
        timeout=10.0)
    node.switch_to_trajectory()
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