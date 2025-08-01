import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

class BoxVisualizer(Node):
    def __init__(self):
        super().__init__('box_tf_and_marker')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        now = self.get_clock().now().to_msg()

        for i in range(6):
            # --- Publish TF ---
            t = TransformStamped()
            
            t.header.stamp = now
            t.header.frame_id = 'world'
            t.child_frame_id = f'object{i}'
            # check if we can lookup the transform
            if not self.tf_buffer.can_transform(t.header.frame_id, t.child_frame_id, now):
                self.get_logger().warn(f"Cannot transform 'world' to 'object{i}' at time {now}")
                continue
            t.transform.translation.x = i * 0.5
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

            # --- Publish Marker ---
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = f'box_{i}'  # Tie marker to the TF frame
            marker.ns = 'box'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0  # identity (already defined by TF)

            # Set box size
            marker.scale.x = 0.030  # Length
            marker.scale.y = 0.120  # Width
            marker.scale.z = 0.032  # Height

            # Set color (RGBA)
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 0.3
            marker.color.a = 1.0

            # marker.lifetime.sec = 1  # Optional: refresh every second
            self.marker_publisher.publish(marker)

def main():
    rclpy.init()
    node = BoxVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()