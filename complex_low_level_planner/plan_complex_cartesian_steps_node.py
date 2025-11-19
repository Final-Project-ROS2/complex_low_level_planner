import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.task import Future

from geometry_msgs.msg import Pose
from custom_interfaces.action import MoveitRelative, GetCurrentPose

import math

class PlanComplexCartesianSteps(Node):
    def __init__(self):
        super().__init__('plan_complex_cartesian_steps_node')

        # Action server
        from custom_interfaces.action import PlanComplexCartesianSteps
        self._action_server = ActionServer(
            self,
            PlanComplexCartesianSteps,
            '/plan_complex_cartesian_steps',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Action clients
        self.get_current_pose_client = ActionClient(self, GetCurrentPose, '/get_current_pose')
        self.plan_relative_client = ActionClient(self, MoveitRelative, '/plan_cartesian_relative')

        self.get_logger().info("✅ plan_complex_cartesian_steps_node started.")

    def goal_callback(self, goal_request):
        self.get_logger().info('🎯 Received goal request for complex cartesian steps.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('⚠️ Received request to cancel the goal.')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('🚀 Executing complex cartesian plan...')
        target_pose = goal_handle.request.target_pose
        
        # --- Step 1: Get current pose ---
        current_pose = await self.get_current_pose()
        if current_pose is None:
            goal_handle.abort()
            self.get_logger().error("❌ Failed to get current pose.")
            return self.make_result(False)
        self.get_logger().info("✅ Got current pose.")
        
        # --- Step 2: Compute relative move needed ---
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        dz = target_pose.position.z - current_pose.position.z
        
        # Compute relative rotation as a quaternion
        # q_relative = q_current^-1 * q_target
        q_current = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        q_target = [
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w
        ]
        
        q_relative = self.quaternion_multiply(self.quaternion_inverse(q_current), q_target)
        
        # Convert the RELATIVE rotation to Euler for the move command
        relative_rpy = self.quaternion_to_euler_from_list(q_relative)
        
        # --- Step 3: Split into multiple single-axis moves ---
        steps = [
            {"dx": dx, "dy": 0.0, "dz": 0.0, "r": 0.0, "p": 0.0, "y": 0.0},
            {"dx": 0.0, "dy": dy, "dz": 0.0, "r": 0.0, "p": 0.0, "y": 0.0},
            {"dx": 0.0, "dy": 0.0, "dz": dz, "r": 0.0, "p": 0.0, "y": 0.0},
            {"dx": 0.0, "dy": 0.0, "dz": 0.0, "r": relative_rpy[0], "p": relative_rpy[1], "y": relative_rpy[2]},
        ]
        
        # --- Step 4: Execute each relative move ---
        for i, step in enumerate(steps):
            if all(abs(v) < 1e-6 for v in step.values()):
                continue  # skip near-zero moves
            self.get_logger().info(f"➡️ Step {i+1}: Moving by {step}")
            success = await self.call_plan_relative(
                step["dx"], step["dy"], step["dz"],
                step["r"], step["p"], step["y"]
            )
            if not success:
                goal_handle.abort()
                self.get_logger().error(f"❌ Step {i+1} failed.")
                return self.make_result(False)
        
        self.get_logger().info("✅ All steps completed successfully.")
        goal_handle.succeed()
        return self.make_result(True)

    async def get_current_pose(self):
        """Call /get_current_pose and return Pose if success."""
        if not self.get_current_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ /get_current_pose action server not available.")
            return None

        goal_msg = GetCurrentPose.Goal()
        goal_future = self.get_current_pose_client.send_goal_async(goal_msg)
        goal_handle = await goal_future

        if not goal_handle.accepted:
            self.get_logger().error("❌ /get_current_pose goal rejected.")
            return None

        result_future = goal_handle.get_result_async()
        result = await result_future

        if not result.result.success:
            self.get_logger().error("❌ /get_current_pose returned unsuccessful result.")
            return None

        return result.result.pose

    async def call_plan_relative(self, dx, dy, dz, roll, pitch, yaw):
        """Call /plan_cartesian_relative once and return success."""
        if not self.plan_relative_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ /plan_cartesian_relative server not available.")
            return False

        goal_msg = MoveitRelative.Goal()
        goal_msg.distance_x = dx
        goal_msg.distance_y = dy
        goal_msg.distance_z = dz
        goal_msg.roll = roll
        goal_msg.pitch = pitch
        goal_msg.yaw = yaw

        goal_future = self.plan_relative_client.send_goal_async(goal_msg)
        goal_handle = await goal_future

        if not goal_handle.accepted:
            self.get_logger().error("❌ /plan_cartesian_relative goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        result = await result_future
        return result.result.success

    def quaternion_to_euler(self, q):
        """Convert quaternion to roll, pitch, yaw."""
        x, y, z, w = q.x, q.y, q.z, q.w
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def quaternion_inverse(self, q):
        """Returns the inverse of quaternion [x, y, z, w]"""
        return [-q[0], -q[1], -q[2], q[3]]

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]

    def quaternion_to_euler_from_list(self, q):
        """Convert quaternion [x, y, z, w] to Euler angles [roll, pitch, yaw]"""
        # Use your existing quaternion_to_euler but adapt for list input
        # Or implement directly:
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]

    def make_result(self, success: bool):
        from custom_interfaces.action import PlanComplexCartesianSteps
        result = PlanComplexCartesianSteps.Result()
        result.success = success
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PlanComplexCartesianSteps()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
