## COPY PASTE THIS 
from rclpy.subscription import Subscription
from CC_RTOLS_V1_0 import StoredData, ModelStructure
from sensor_msgs.msg import Imu
from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
import mavros

def euler_from_quaternion(x:float, y:float, z:float, w:float) -> tuple:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('OLSNode')
        
        self.ols_publisher: Publisher = self.create_publisher(
            Float64MultiArray, 'ols_data', 10)
        
        # imu
        self.imu_subscription: Subscription = self.create_subscription(
            Imu, '/mavros/imu/data', self.imu_callback, SENSOR_QOS)
        
        # mavros state
        self.state_sub: Subscription = self.create_subscription(
            mavros.local_position.Odometry,
            'mavros/local_position/odom',
            self.mavros_state_callback,
            qos_profile=SENSOR_QOS)
        
        # rc channel
        self.rc_sub: Subscription = self.create_subscription(
            RCOut,
            '/mavros/rc/out',
            self.rc_callback,
            qos_profile=SENSOR_QOS)
        
        # XANDER look at group subscription at same time 
        
        # ros2 topic echo /mavros/rc/out
        self.timer_period: float = 0.1  # seconds some frequency
        self.timer = self.create_timer(
            self.timer_period, self.publish_ols_data)

        # Publishing the values -> so that way we can do post processing on the data
        self.rol = ModelStructure(2)
        self.pit = ModelStructure(2)
        self.yaw = ModelStructure(2)

        self.current_state: List[float] = [
                None,  # x
                None,  # y
                None,  # z
                None,  # phi
                None,  # theta
                None,  # psi
                None,  # airspeed
            ]

    def rc_callback(self, msg: RCOut) -> None:
        print(f"Received RC data: {msg.channels}")

    def mavros_state_callback(self, msg: mavros.local_position.Odometry) -> None:
        """
        Converts NED to ENU and publishes the trajectory
        https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
        Twist Will show velocity in linear and rotational 
        """
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        self.current_state[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = euler_from_quaternion(
            qx, qy, qz, qw)

        self.current_state[3] = roll
        self.current_state[4] = pitch
        self.current_state[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        # get magnitude of velocity
        self.current_state[6] = np.sqrt(vx**2 + vy**2 + vz**2)

    def imu_callback(self, msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        self.get_logger().info(f"Received: {msg}")
        # line_2: str = "I was wondering if after all these years you'd like to meet"
        # self.get_logger().info(f"Sending: {line_2}")

    def publish_ols_data(self) -> None:
        ## Change *_acel data to the values from current state or whatever you want
        self.rol.update_model(self.rol_accel.data[0], ([self.rol_velo.data[1] * (CompanionComputer.b / (2 * np.mean(self.airspeed.data))), self.ail_def.data[0]] * ((self.dyn_pres.data[0] * CompanionComputer.S * CompanionComputer.b) / CompanionComputer.Ix)))
        self.pit.update_model(self.pit_accel.data[0], [self.pit_velo.data[1], self.elv_def.data[0]])
        self.yaw.update_model(self.yaw_accel.data[0], [self.yaw_velo.data[1], self.rud_def.data[0]])
    
 