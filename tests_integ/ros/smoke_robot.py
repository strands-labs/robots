"""A real rclpy 'robot' for the smoke test: ros2_control-style topics, BEST_EFFORT joint states, latched URDF."""

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_msgs.msg import TFMessage

_LIMIT = '<limit lower="-2" upper="2" effort="1" velocity="1"/>'
_LINKS = "".join(
    f'<link name="l{i}"><visual><geometry><mesh filename="package://smoke/l{i}.stl"/></geometry></visual></link>'
    f'<joint name="{n}" type="revolute"><parent link="base"/><child link="l{i}"/>{_LIMIT}</joint>'
    for i, n in enumerate(["shoulder", "elbow", "wrist"])
)
URDF = f'<robot name="smoke_arm"><link name="base"/>{_LINKS}</robot>'


class Arm(Node):
    def __init__(self):
        super().__init__("smoke_arm")
        self.declare_parameter("speed_scale", 0.5)
        self.q = {"wrist": 0.3, "shoulder": 0.1, "elbow": -0.2}
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.js = self.create_publisher(JointState, "/joint_states", be)
        self.create_publisher(String, "/robot_description", latch).publish(String(data=URDF))
        t = TransformStamped()
        t.header.frame_id = "base"
        t.child_frame_id = "tool"
        t.transform.translation.x = 0.4
        t.transform.rotation.w = 1.0
        self.create_publisher(TFMessage, "/tf_static", latch).publish(TFMessage(transforms=[t]))
        self.got = self.create_publisher(String, "/smoke/received", 10)
        self.create_subscription(Float64MultiArray, "/forward_position_controller/commands", self.on_cmd, 10)
        self.create_timer(0.01, self.tick)
        self.n = 0

    def tick(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(self.q)
        m.position = list(self.q.values())
        m.velocity = [0.0] * 3
        m.effort = [0.0] * 3
        self.js.publish(m)

    def on_cmd(self, msg):
        self.n += 1
        for n, v in zip(["shoulder", "elbow", "wrist"], msg.data, strict=False):
            self.q[n] = v
        self.got.publish(String(data=f"{self.n} {list(msg.data)}"))


rclpy.init()
node = Arm()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
