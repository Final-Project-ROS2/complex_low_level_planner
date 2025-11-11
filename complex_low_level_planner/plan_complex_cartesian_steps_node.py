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

        # Convert quaternion to Euler
        current_rpy = self.quaternion_to_euler(current_pose.orientation)
        target_rpy = self.quaternion_to_euler(target_pose.orientation)
        droll = target_rpy[0] - current_rpy[0]
        dpitch = target_rpy[1] - current_rpy[1]
        dyaw = target_rpy[2] - current_rpy[2]

        # --- Step 3: Split into multiple single-axis moves ---
        steps = [
            {"dx": dx, "dy": 0.0, "dz": 0.0, "r": 0.0, "p": 0.0, "y": 0.0},
            {"dx": 0.0, "dy": dy, "dz": 0.0, "r": 0.0, "p": 0.0, "y": 0.0},
            {"dx": 0.0, "dy": 0.0, "dz": dz, "r": 0.0, "p": 0.0, "y": 0.0},
            {"dx": 0.0, "dy": 0.0, "dz": 0.0, "r": droll, "p": dpitch, "y": dyaw},
            # {"dx": 0.0, "dy": 0.0, "dz": 0.0, "r": 0.0, "p": dpitch, "y": 0.0},
            # {"dx": 0.0, "dy": 0.0, "dz": 0.0, "r": 0.0, "p": 0.0, "y": dyaw},
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
