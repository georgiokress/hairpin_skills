#!/usr/bin/env python3
"""Publish ATI Net F/T measurements through the documented TCP interface.

The node uses only READCALINFO and READFT on TCP port 49151. The READFT
request has both system-command fields set to zero, so it cannot bias the
sensor, clear latches, change configuration, or affect robot motion. This TCP
channel can run in parallel with the ATI URCap's single RDT/UDP connection.
"""

from __future__ import annotations

import math
import socket
import struct
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node


READ_FT_REQUEST = bytes(20)
READ_CALIBRATION_REQUEST = bytes([1]) + bytes(19)
CALIBRATION_RESPONSE_SIZE = 24
WRENCH_RESPONSE_SIZE = 16


class AtiNetftTcpWrench(Node):
    """Read and publish validated ATI samples without taking over RDT."""

    def __init__(self) -> None:
        super().__init__("ati_netft_tcp_wrench")

        self.declare_parameter("sensor_ip", "192.168.214.70")
        self.declare_parameter("tcp_port", 49151)
        # Net F/T configuration #1 already rotates the configured wrench into
        # the UR/tool orientation.  The point of application remains at the
        # ATI measuring origin, 76 mm from tool0, represented by
        # ati_ur_aligned.  Labelling this data ati_sensor would apply the
        # configuration rotation a second time in downstream TF transforms.
        self.declare_parameter("frame_id", "ati_ur_aligned")
        self.declare_parameter("topic", "/ati/wrench_tcp")
        self.declare_parameter("poll_period_s", 0.002)
        self.declare_parameter("socket_timeout_s", 0.05)
        self.declare_parameter("reconnect_period_s", 0.20)

        self.sensor_ip = str(self.get_parameter("sensor_ip").value)
        self.tcp_port = int(self.get_parameter("tcp_port").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        topic = str(self.get_parameter("topic").value)
        poll_period = float(self.get_parameter("poll_period_s").value)
        self.socket_timeout = float(
            self.get_parameter("socket_timeout_s").value
        )
        self.reconnect_period = float(
            self.get_parameter("reconnect_period_s").value
        )
        if not 1 <= self.tcp_port <= 65535:
            raise ValueError("tcp_port must be in [1, 65535].")
        if min(poll_period, self.socket_timeout, self.reconnect_period) <= 0.0:
            raise ValueError("TCP timing parameters must be > 0.")

        self.publisher = self.create_publisher(WrenchStamped, topic, 500)
        self.connection: Optional[socket.socket] = None
        self.scales: Optional[list[float]] = None
        self.last_connect_attempt = 0.0
        self.last_error_log = 0.0
        self.first_sample_published = False
        self.sample_count = 0
        self.rate_window_start = time.monotonic()
        self.timer = self.create_timer(poll_period, self.poll)

        self.get_logger().info(
            "ATI Net F/T read-only TCP client configured for "
            f"{self.sensor_ip}:{self.tcp_port}; publishing {topic} at up to "
            f"{1.0 / poll_period:.1f} Hz with frame_id='{self.frame_id}'."
        )

    def _log_error_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_error_log >= 2.0:
            self.get_logger().error(message)
            self.last_error_log = now

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < size:
            chunk = connection.recv(size - received)
            if not chunk:
                raise ConnectionError(
                    f"ATI TCP connection closed after {received}/{size} bytes."
                )
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _disconnect(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except OSError:
                pass
        self.connection = None
        self.scales = None

    def _connect(self) -> None:
        connection = socket.create_connection(
            (self.sensor_ip, self.tcp_port),
            timeout=self.socket_timeout,
        )
        connection.settimeout(self.socket_timeout)
        connection.sendall(READ_CALIBRATION_REQUEST)
        response = self._recv_exact(connection, CALIBRATION_RESPONSE_SIZE)
        header, force_units, torque_units, counts_force, counts_torque, *raw_scales = (
            struct.unpack("!HBBII6H", response)
        )
        if header != 0x1234:
            connection.close()
            raise RuntimeError(
                f"ATI calibration response has invalid header 0x{header:04X}."
            )
        if force_units != 2 or torque_units != 3:
            connection.close()
            raise RuntimeError(
                "ATI TCP units must be N and Nm; received "
                f"force_units={force_units}, torque_units={torque_units}."
            )
        if counts_force <= 0 or counts_torque <= 0:
            connection.close()
            raise RuntimeError("ATI calibration returned invalid counts per unit.")

        self.scales = [
            raw_scales[index]
            / float(counts_force if index < 3 else counts_torque)
            for index in range(6)
        ]
        if not all(value > 0.0 and math.isfinite(value) for value in self.scales):
            connection.close()
            self.scales = None
            raise RuntimeError("ATI calibration returned invalid scale factors.")
        self.connection = connection
        self.get_logger().info(
            "ATI TCP connected; force LSB="
            f"{[round(value, 9) for value in self.scales[:3]]} N, "
            "torque LSB="
            f"{[round(value, 9) for value in self.scales[3:]]} Nm."
        )

    def _read_wrench(self) -> list[float]:
        if self.connection is None or self.scales is None:
            raise RuntimeError("ATI TCP connection is not initialized.")
        self.connection.sendall(READ_FT_REQUEST)
        response = self._recv_exact(self.connection, WRENCH_RESPONSE_SIZE)
        header, status, *counts = struct.unpack("!HH6h", response)
        if header != 0x1234:
            raise RuntimeError(
                f"ATI wrench response has invalid header 0x{header:04X}."
            )
        if status != 0:
            raise RuntimeError(
                f"ATI wrench response reports status 0x{status:04X}."
            )
        return [
            float(counts[index]) * self.scales[index]
            for index in range(6)
        ]

    def poll(self) -> None:
        now = time.monotonic()
        if self.connection is None:
            if now - self.last_connect_attempt < self.reconnect_period:
                return
            self.last_connect_attempt = now
            try:
                self._connect()
            except (OSError, RuntimeError, struct.error) as exc:
                self._disconnect()
                self._log_error_throttled(f"ATI TCP connection failed: {exc}")
                return

        try:
            values = self._read_wrench()
        except (OSError, RuntimeError, struct.error) as exc:
            self._disconnect()
            self._log_error_throttled(f"ATI TCP read failed: {exc}")
            return

        message = WrenchStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.wrench.force.x = values[0]
        message.wrench.force.y = values[1]
        message.wrench.force.z = values[2]
        message.wrench.torque.x = values[3]
        message.wrench.torque.y = values[4]
        message.wrench.torque.z = values[5]
        self.publisher.publish(message)

        self.sample_count += 1
        if not self.first_sample_published:
            self.first_sample_published = True
            self.get_logger().info(
                "First valid ATI TCP wrench published; READFT system-command "
                "bits are zero and no sensor-control command was issued."
            )
        elapsed = now - self.rate_window_start
        if elapsed >= 5.0:
            self.get_logger().info(
                f"ATI TCP publish rate: {self.sample_count / elapsed:.1f} Hz."
            )
            self.sample_count = 0
            self.rate_window_start = now

    def destroy_node(self):
        self._disconnect()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AtiNetftTcpWrench()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
