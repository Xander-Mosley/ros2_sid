#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node

from mavros_msgs.msg import State, ParamValue
from mavros_msgs.srv import ParamPull, ParamGet

MAVROS_NS = "/mavros"         # adjust if you run a namespace like "/uav1/mavros"
PARAM_NAME = "SID_TYPE"       # change to the FCU parameter you want to read

def srv(ns: str, tail: str) -> str:
    return f"{ns.rstrip('/')}/param/{tail}"

class PullAndGet(Node):
    def __init__(self, ns=MAVROS_NS):
        super().__init__("pull_and_get_demo")
        self.ns = ns
        self.connected = False

        # Subscribe to MAVROS connection state
        self.create_subscription(State, f"{ns}/state", self._state_cb, 10)

        # Create service clients
        self.cli_pull = self.create_client(ParamPull, srv(ns, "pull"))
        self.cli_get  = None  # we'll create it after we confirm /param/get exists

    def _state_cb(self, msg: State):
        self.connected = bool(msg.connected)

    def wait_for_connected(self, total=60.0) -> bool:
        t0 = time.time()
        self.get_logger().info("Waiting for MAVROS to report connected...")
        while rclpy.ok() and (time.time() - t0) < total:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.connected:
                self.get_logger().info("MAVROS connected ✓")
                return True
        self.get_logger().error("Timed out waiting for MAVROS connection.")
        return False

    def wait_for_service(self, client, name: str, total=30.0) -> bool:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < total:
            if client.wait_for_service(timeout_sec=1.0):
                return True
            self.get_logger().info(f"Waiting for {name} ...")
        self.get_logger().error(f"Service not available: {name}")
        return False

    def pull_params(self, timeout=20.0) -> bool:
        if not self.wait_for_service(self.create_client(ParamPull, srv(self.ns, "pull")),
                                     srv(self.ns, "pull")):
            return False
        req = ParamPull.Request()
        req.force_pull = True
        fut = self.cli_pull.call_async(req)
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if fut.done():
                res = fut.result()
                ok = bool(res and res.success)
                print("res:", res)
                self.get_logger().info(f"Param pull success={ok}, received={getattr(res, 'param_received', 0)}")
                return ok
        self.get_logger().error("Param pull timed out.")
        return False

    def wait_for_param_get_to_appear(self, total=30.0) -> bool:
        """Some setups only advertise /param/get after connection; wait for it to show up."""
        want_name = srv(self.ns, "get")
        want_type = "mavros_msgs/srv/ParamGet"
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < total:
            for name, types in self.get_service_names_and_types():
                if name == want_name and want_type in types:
                    self.get_logger().info("Found /mavros/param/get ✓")
                    self.cli_get = self.create_client(ParamGet, want_name)
                    return True
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().error("'/mavros/param/get' never appeared (param plugin not ready/filtered?).")
        return False

    def get_param(self, key: str, timeout=10.0):
        req = ParamGet.Request()
        req.param_id = key
        fut = self.cli_get.call_async(req)
        t0 = time.time()
        self.get_logger().info(f"Requesting parameter: {key}")
        while rclpy.ok() and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if fut.done():
                res = fut.result()
                if not res or not res.success:
                    raise RuntimeError(f"Get failed for {key}")
                pv: ParamValue = res.value
                return int(pv.integer) if abs(pv.real - round(pv.real)) < 1e-6 else float(pv.real)
        raise TimeoutError(f"ParamGet timed out for {key}")

def main():
    rclpy.init()
    n = PullAndGet(MAVROS_NS)

    # 1) Wait until MAVROS is connected to the FCU
    if not n.wait_for_connected(total=60.0):
        n.destroy_node(); rclpy.shutdown(); return

    # 2) Pull the whole param table (refresh cache)
    n.pull_params(timeout=20.0)

    # 3) Wait for /mavros/param/get to appear, then create the client
    if not n.wait_for_param_get_to_appear(total=30.0):
        n.get_logger().error("ParamGet not available; check plugin allow/deny lists or environment.")
        n.destroy_node(); rclpy.shutdown(); return

    # 4) Read the parameter
    try:
        value = n.get_param(PARAM_NAME, timeout=10.0)
        n.get_logger().info(f"{PARAM_NAME} = {value}")
    except Exception as e:
        n.get_logger().error(str(e))

    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
