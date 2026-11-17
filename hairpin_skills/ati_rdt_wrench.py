#!/usr/bin/env python3

import socket
import struct
import threading
import time

from geometry_msgs.msg import WrenchStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class AtiRdtWrench(Node):
    """Read ATI Net F/T RDT data and publish it as WrenchStamped."""

    def __init__(self):
        super().__init__("ati_rdt_wrench")

        self.declare_parameter("standalone_mode_confirmed", False)
        if not bool(self.get_parameter("standalone_mode_confirmed").value):
            raise RuntimeError(
                "ati_rdt_wrench is locked because Net F/T supplies a single "
                "RDT destination. Starting it while ATI URCap is active "
                "removes the URCap sensor stream. Use the read-only URCap "
                "measurement path, or explicitly set "
                "standalone_mode_confirmed:=true only when ATI URCap is "
                "disabled."
            )

        self.get_logger().warning(
            "Standalone RDT mode explicitly confirmed. ATI URCap must remain "
            "disabled until this node has stopped."
        )

        self.declare_parameter("sensor_ip", "192.168.214.70")
        self.declare_parameter("sensor_port", 49152)
        self.declare_parameter("socket_receive_buffer_bytes", 4_194_304)
        self.declare_parameter("publish_rate_hz", 500.0)

        self.declare_parameter("counts_per_force", 1_000_000.0)
        self.declare_parameter("counts_per_torque", 1_000_000.0)

        # NET F/T configuration #1 rotates the RDT wrench into the UR/tool0
        # orientation.  Its point of application remains the ATI measuring
        # origin, 76 mm from tool0, so it must not be labelled ati_sensor
        # (which denotes the unrotated physical sensor axes).
        self.declare_parameter("frame_id", "ati_ur_aligned")
        self.declare_parameter("topic", "/ati/wrench_raw")

        self.sensor_ip = (
            self.get_parameter("sensor_ip")
            .get_parameter_value()
            .string_value
        )

        self.sensor_port = (
            self.get_parameter("sensor_port")
            .get_parameter_value()
            .integer_value
        )

        self.socket_receive_buffer_bytes = int(
            self.get_parameter("socket_receive_buffer_bytes").value
        )
        if self.socket_receive_buffer_bytes < 65_536:
            raise ValueError(
                "socket_receive_buffer_bytes must be at least 65536."
            )

        self.publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0.")

        self.counts_per_force = (
            self.get_parameter("counts_per_force")
            .get_parameter_value()
            .double_value
        )

        self.counts_per_torque = (
            self.get_parameter("counts_per_torque")
            .get_parameter_value()
            .double_value
        )

        self.frame_id = (
            self.get_parameter("frame_id")
            .get_parameter_value()
            .string_value
        )

        topic = (
            self.get_parameter("topic")
            .get_parameter_value()
            .string_value
        )

        wrench_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.publisher = self.create_publisher(
            WrenchStamped,
            topic,
            wrench_qos,
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            self.socket_receive_buffer_bytes,
        )
        self.actual_socket_receive_buffer_bytes = self.sock.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
        )
        self.sock.settimeout(0.10)

        self.sensor_address = (
            self.sensor_ip,
            self.sensor_port,
        )

        # ATI RDT request:
        # uint16 header      = 0x1234
        # uint16 command     = 0x0002
        # uint32 sampleCount = 0 (continuous)
        self.start_request = struct.pack(
            ">HHI",
            0x1234,
            0x0002,
            0,
        )

        # Stop streaming command
        self.stop_request = struct.pack(
            ">HHI",
            0x1234,
            0x0000,
            0,
        )

        self.last_rdt_sequence = None
        self.dropped_packets = 0
        self.dropped_packets_since_log = 0
        self.last_drop_log = 0.0
        self.received_packets = 0
        self.rate_window_start = time.monotonic()
        self.first_packet_received = False
        self.first_packet_published = False
        self.stop_event = threading.Event()
        self.destroyed = False
        self.latest_lock = threading.Lock()
        self.latest_message = None
        self.latest_sequence = None
        self.last_published_sequence = None

        self.sock.sendto(
            self.start_request,
            self.sensor_address,
        )

        self.get_logger().info(
            f"ATI RDT streaming started from "
            f"{self.sensor_ip}:{self.sensor_port}"
        )

        self.get_logger().info(
            f"Publishing ATI wrench on {topic}, "
            f"frame_id='{self.frame_id}'"
        )

        self.get_logger().info(
            "ATI RDT receiver uses a dedicated blocking thread; "
            f"UDP receive buffer={self.actual_socket_receive_buffer_bytes} bytes."
        )

        # A dedicated blocking receiver drains the UDP socket continuously.
        # A ROS timer is not deterministic enough for the sensor's rounded-up
        # RDT rate (roughly 760 Hz on this unit).
        self.receiver_thread = threading.Thread(
            target=self.receive_loop,
            name="ati_rdt_receiver",
            daemon=True,
        )
        self.receiver_thread.start()

        # Keep UDP reception independent of potentially blocking reliable DDS
        # publication. The receiver always drains the sensor socket; the ROS
        # executor publishes the newest complete sample at the configured rate.
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_latest,
        )

    def receive_loop(self):
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(1024)

            except socket.timeout:
                continue

            except OSError as exc:
                if not self.stop_event.is_set() and rclpy.ok():
                    self.get_logger().error(
                        f"UDP receive failed: {exc}"
                    )
                break

            self.process_packet(data)

    def process_packet(self, data):

        if len(data) < 36:
            self.get_logger().warning(
                f"Unexpected RDT packet length: {len(data)} bytes"
            )
            return

        try:
            (
                rdt_sequence,
                ft_sequence,
                status,
                fx_counts,
                fy_counts,
                fz_counts,
                tx_counts,
                ty_counts,
                tz_counts,
            ) = struct.unpack(
                ">IIIiiiiii",
                data[:36],
            )

        except struct.error as exc:
            self.get_logger().error(
                f"Could not decode RDT packet: {exc}"
            )
            return

        if self.last_rdt_sequence is not None:
            expected = (self.last_rdt_sequence + 1) & 0xFFFFFFFF

            if rdt_sequence != expected:
                difference = (
                    rdt_sequence - expected
                ) & 0xFFFFFFFF

                self.dropped_packets += difference
                self.dropped_packets_since_log += difference
                now = time.monotonic()
                if now - self.last_drop_log >= 2.0 and rclpy.ok():
                    self.get_logger().warning(
                        "RDT sequence gap(s): "
                        f"latest expected={expected}, got={rdt_sequence}, "
                        f"lost_since_last_log={self.dropped_packets_since_log}, "
                        f"lost_total={self.dropped_packets}"
                    )
                    self.dropped_packets_since_log = 0
                    self.last_drop_log = now

        self.last_rdt_sequence = rdt_sequence

        if status != 0:
            self.get_logger().warning(
                f"ATI system status != 0: 0x{status:08X}"
            )

        fx = fx_counts / self.counts_per_force
        fy = fy_counts / self.counts_per_force
        fz = fz_counts / self.counts_per_force

        tx = tx_counts / self.counts_per_torque
        ty = ty_counts / self.counts_per_torque
        tz = tz_counts / self.counts_per_torque

        msg = WrenchStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.wrench.force.x = float(fx)
        msg.wrench.force.y = float(fy)
        msg.wrench.force.z = float(fz)

        msg.wrench.torque.x = float(tx)
        msg.wrench.torque.y = float(ty)
        msg.wrench.torque.z = float(tz)

        with self.latest_lock:
            self.latest_message = msg
            self.latest_sequence = rdt_sequence

        self.received_packets += 1
        if not self.first_packet_received:
            self.first_packet_received = True
            self.get_logger().info(
                "First valid ATI RDT packet received."
            )
        now = time.monotonic()
        elapsed = now - self.rate_window_start
        if elapsed >= 5.0:
            self.get_logger().info(
                f"ATI RDT receive rate: "
                f"{self.received_packets / elapsed:.1f} Hz; "
                f"lost_total={self.dropped_packets}."
            )
            self.received_packets = 0
            self.rate_window_start = now

    def publish_latest(self):
        with self.latest_lock:
            if (
                self.latest_message is None
                or self.latest_sequence == self.last_published_sequence
            ):
                return
            msg = self.latest_message
            sequence = self.latest_sequence

        self.publisher.publish(msg)
        self.last_published_sequence = sequence
        if not self.first_packet_published:
            self.first_packet_published = True
            self.get_logger().info(
                "First valid ATI RDT packet published."
            )

    def destroy_node(self):
        if self.destroyed:
            return
        self.destroyed = True
        self.stop_event.set()

        try:
            self.sock.sendto(
                self.stop_request,
                self.sensor_address,
            )
        except OSError:
            pass

        try:
            self.sock.close()
        except OSError:
            pass

        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)

        if rclpy.ok():
            self.get_logger().info(
                "ATI RDT streaming stopped; "
                f"estimated lost_total={self.dropped_packets}."
            )

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = AtiRdtWrench()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
