#!/usr/bin/env python3

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from ur_msgs.srv import SetForceMode


class ForceModeIsolated(Node):
    """Runs and records one isolated normal-force experiment."""

    def __init__(self) -> None:
        super().__init__("force_mode_isolated")

        # Experiment parameters
        self.declare_parameter("bag_name", "force_mode_isolated")
        self.declare_parameter(
            "bag_root",
            str(Path.home() / "masterarbeit" / "bags"),
        )
        self.declare_parameter("task_frame", "sliding_frame")
        self.declare_parameter("force_z", -1.0)
        self.declare_parameter("hold_time", 2.0)
        self.declare_parameter("speed_limit_z", 0.005)
        self.declare_parameter("damping_factor", 0.5)
        self.declare_parameter("gain_scaling", 0.3)
        self.declare_parameter("tare_wait_time", 1.0)
        self.declare_parameter("bag_start_wait_time", 2.0)
        self.declare_parameter("interactive", True)

        # ROS interfaces
        self.zero_client = self.create_client(
            Trigger,
            "/io_and_status_controller/zero_ftsensor",
        )
        self.start_force_client = self.create_client(
            SetForceMode,
            "/force_mode_controller/start_force_mode",
        )
        self.stop_force_client = self.create_client(
            Trigger,
            "/force_mode_controller/stop_force_mode",
        )

        self.bag_process: Optional[subprocess.Popen] = None
        self.force_mode_active = False

    def parameter(self, name: str):
        return self.get_parameter(name).value

    def wait_for_services(self) -> None:
        services = (
            (self.zero_client, "zero_ftsensor"),
            (self.start_force_client, "start_force_mode"),
            (self.stop_force_client, "stop_force_mode"),
        )

        for client, name in services:
            self.get_logger().info(f"Warte auf {name} ...")
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service nicht verfügbar: {name}")

    def call_trigger(self, client, name: str) -> None:
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None:
            raise RuntimeError(f"Keine Antwort von {name}")

        if not result.success:
            raise RuntimeError(
                f"{name} fehlgeschlagen: {result.message}"
            )

        self.get_logger().info(
            f"{name} erfolgreich: {result.message}"
        )

    def zero_sensor(self) -> None:
        self.get_logger().info("Kraft-Momenten-Sensor nullen ...")
        self.call_trigger(self.zero_client, "zero_ftsensor")
        time.sleep(float(self.parameter("tare_wait_time")))

    def make_force_request(self) -> SetForceMode.Request:
        request = SetForceMode.Request()

        request.task_frame.header.frame_id = str(
            self.parameter("task_frame")
        )
        request.task_frame.pose.orientation.w = 1.0

        request.selection_vector_x = False
        request.selection_vector_y = False
        request.selection_vector_z = True
        request.selection_vector_rx = False
        request.selection_vector_ry = False
        request.selection_vector_rz = False

        request.wrench.force.z = float(self.parameter("force_z"))

        # UR Force Mode type 2: frame is not transformed dynamically.
        request.type = 2

        request.speed_limits.linear.z = float(
            self.parameter("speed_limit_z")
        )

        request.deviation_limits = [
            0.002,
            0.002,
            0.005,
            0.05,
            0.05,
            0.05,
        ]

        request.damping_factor = float(
            self.parameter("damping_factor")
        )
        request.gain_scaling = float(
            self.parameter("gain_scaling")
        )

        return request

    def start_force_mode(self) -> None:
        force_z = float(self.parameter("force_z"))
        task_frame = str(self.parameter("task_frame"))

        self.get_logger().info(
            f"Starte Force Mode: Fz={force_z:.3f} N "
            f"im Frame {task_frame}"
        )

        future = self.start_force_client.call_async(
            self.make_force_request()
        )
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None:
            raise RuntimeError("Keine Antwort von start_force_mode")

        if not result.success:
            raise RuntimeError("start_force_mode meldet success=False")

        self.force_mode_active = True
        self.get_logger().info("Force Mode aktiv.")

    def stop_force_mode(self) -> None:
        if not self.force_mode_active:
            return

        self.get_logger().info("Stoppe Force Mode ...")
        self.call_trigger(
            self.stop_force_client,
            "stop_force_mode",
        )
        self.force_mode_active = False

    def get_bag_path(self) -> Path:
        bag_root = Path(str(self.parameter("bag_root"))).expanduser()
        bag_root.mkdir(parents=True, exist_ok=True)

        bag_name = str(self.parameter("bag_name")).strip()
        if not bag_name:
            raise ValueError("Parameter bag_name darf nicht leer sein.")

        return bag_root / bag_name

    def start_bag(self, bag_path: Path) -> None:
        if bag_path.exists():
            raise FileExistsError(
                f"Bag existiert bereits: {bag_path}"
            )

        command = [
            "ros2",
            "bag",
            "record",
            "-s",
            "mcap",
            "/tf",
            "/tf_static",
            "/joint_states",
            "/force_torque_sensor_broadcaster/wrench",
            "-o",
            str(bag_path),
        ]

        self.get_logger().info(f"Starte Bag: {bag_path}")

        self.bag_process = subprocess.Popen(
            command,
            start_new_session=True,
        )

        time.sleep(float(self.parameter("bag_start_wait_time")))

        if self.bag_process.poll() is not None:
            raise RuntimeError(
                "ros2 bag record wurde unerwartet beendet."
            )

    def stop_bag(self) -> None:
        if self.bag_process is None:
            return

        if self.bag_process.poll() is None:
            self.get_logger().info("Stoppe Bag-Aufnahme ...")

            # Signal an die komplette Prozessgruppe, damit rosbag2
            # seine Metadaten und die MCAP-Datei sauber abschließt.
            try:
                pgid = self.bag_process.pid
                import os
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass

            try:
                self.bag_process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self.get_logger().warning(
                    "Bag reagiert nicht auf SIGINT; sende SIGTERM."
                )
                try:
                    import os
                    os.killpg(self.bag_process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.bag_process.wait(timeout=5.0)

        self.bag_process = None
        self.get_logger().info("Bag-Aufnahme beendet.")

    def wait_for_enter(self, message: str) -> None:
        if bool(self.parameter("interactive")):
            input(message)

    def run(self) -> None:
        self.wait_for_services()
        bag_path = self.get_bag_path()

        self.get_logger().info(
            "Versuchsablauf: Positionieren -> Tare -> Bag -> "
            "Force Mode -> Halten -> Stop -> Rückzug -> Bag Stop"
        )

        self.wait_for_enter(
            "\nTCP manuell 1–2 mm über der Oberfläche positionieren.\n"
            "Kein Kontakt; Roboter stillhalten.\n"
            "Danach Enter drücken ..."
        )

        self.zero_sensor()
        self.start_bag(bag_path)

        try:
            self.start_force_mode()

            hold_time = float(self.parameter("hold_time"))
            self.get_logger().info(
                f"Kontakt herstellen und {hold_time:.3f} s halten ..."
            )
            time.sleep(hold_time)

            self.stop_force_mode()

            self.wait_for_enter(
                "\nForce Mode wurde gestoppt.\n"
                "TCP jetzt manuell von der Oberfläche zurückziehen.\n"
                "Danach Enter drücken, um die Bag zu stoppen ..."
            )

        finally:
            # Sicherheitsreihenfolge: zuerst Force Mode beenden,
            # anschließend Bag finalisieren.
            try:
                self.stop_force_mode()
            finally:
                self.stop_bag()

        self.get_logger().info(
            f"Versuch abgeschlossen: {bag_path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForceModeIsolated()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning(
            "Versuch durch Benutzer abgebrochen."
        )
    except Exception as exc:
        node.get_logger().error(f"Versuch fehlgeschlagen: {exc}")
        raise
    finally:
        try:
            node.stop_force_mode()
        except Exception as exc:
            node.get_logger().error(
                f"Force Mode konnte beim Aufräumen nicht gestoppt werden: {exc}"
            )

        try:
            node.stop_bag()
        except Exception as exc:
            node.get_logger().error(
                f"Bag konnte beim Aufräumen nicht gestoppt werden: {exc}"
            )

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
