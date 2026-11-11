from ultralytics import YOLO
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import numpy as np
import cv2
import sys

# RGB-only detection (no depth sensor): runs continuously on every received frame
# The shelf position is recovered from its known height, assuming it stands upright and parallel to the camera's vertical axis
# The orientation cannot be observed from a bounding box: the published pose has identity orientation (camera optical frame)
# Usage: python yolov8/ros_script_rgb_only.py [image_topic] [weights]

classesDict = dict()
classesDict[0] = 'shelf'

rgb_image_topic = sys.argv[1] if len(sys.argv) > 1 else '/webrtc/image_raw'
weights = sys.argv[2] if len(sys.argv) > 2 else 'yolov8/trained_models/shelf/best.pt'
camera_info_topic = rgb_image_topic.rsplit('/', 1)[0] + '/camera_info'
confidence_threshold = 0.3
shelf_height = 1.5 #meters
border_margin = 2 #pixels, a box this close to the top/bottom of the image is considered truncated

output_topic_prefix = '/object_detection/'
pose_topic_prefix = '/object_pose_initialization/'

bridge = CvBridge()
model = YOLO(weights)
node = None
pub_image = None
pub_boxes = None
pub_dict = dict()
camera_info = None

def camera_info_callback(data):
    global camera_info
    camera_info = data

def pixel_to_ray(u, v):
    #undistort the pixel and return its normalized coordinates (x/z, y/z)
    K = np.array(camera_info.k).reshape(3, 3)
    D = np.array(camera_info.d)
    point = np.array([[[u, v]]], dtype=np.float64)
    if camera_info.distortion_model == 'equidistant':
        x, y = cv2.fisheye.undistortPoints(point, K, D[:4]).ravel()
    else:
        x, y = cv2.undistortPoints(point, K, D).ravel()
    return x, y

def estimate_position(x1, y1, x2, y2):
    #the top and bottom edges of the box are the top and bottom of the shelf: their vertical
    #distance in normalized coordinates is shelf_height / Z. Returns (X, Y, Z) of the shelf center in meters
    u = (x1 + x2) / 2
    x_top, y_top = pixel_to_ray(u, y1)
    x_bottom, y_bottom = pixel_to_ray(u, y2)
    if not y_bottom - y_top > 0:
        return None
    Z = shelf_height / (y_bottom - y_top)
    return (x_top + x_bottom) / 2 * Z, (y_top + y_bottom) / 2 * Z, Z

def rgb_callback(data):
    try:
        rgb_data = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    results = model(rgb_data, conf=confidence_threshold, verbose=False)[0]

    #one line per detected object: class confidence x1 y1 x2 y2 (pixels)
    detected_objects_txt = ''
    for cls, conf, xyxy in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
        objectClass = classesDict.get(int(cls.item()))
        if objectClass is None:
            print('Warning: object class not found in the dictionary. Check that the model was trained with the same classes as the dictionary.')
            continue
        x1, y1, x2, y2 = xyxy.cpu().numpy()
        detected_objects_txt += f'{objectClass} {conf.item():.2f} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}\n'

        if camera_info is None:
            node.get_logger().warn(f'No message on {camera_info_topic} yet, pose not published', throttle_duration_sec=5.0)
            continue
        if y1 <= border_margin or y2 >= data.height - 1 - border_margin:
            node.get_logger().warn(f'{objectClass} cut by the top/bottom of the image, pose not published', throttle_duration_sec=5.0)
            continue
        position = estimate_position(x1, y1, x2, y2)
        if position is None:
            node.get_logger().warn(f'Invalid back-projection of the {objectClass} box, pose not published', throttle_duration_sec=5.0)
            continue
        X, Y, Z = position
        pose = PoseStamped()
        pose.header = data.header
        pose.pose.position.x = float(X)
        pose.pose.position.y = float(Y)
        pose.pose.position.z = float(Z)
        pose.pose.orientation.w = 1.0
        pub_dict[int(cls.item())].publish(pose)
        node.get_logger().info(f'{objectClass} position: X={X:.2f} m, Y={Y:.2f} m, Z={Z:.2f} m')
    if detected_objects_txt:
        node.get_logger().info('Detected:\n' + detected_objects_txt.strip())

    msg_str = String()
    msg_str.data = detected_objects_txt
    pub_boxes.publish(msg_str)

    #annotated image, to visualize with rqt_image_view
    image_msg = bridge.cv2_to_imgmsg(results.plot(), "bgr8")
    image_msg.header = data.header
    pub_image.publish(image_msg)


def listener():
    global node, pub_image, pub_boxes, pub_dict
    rclpy.init()
    node = rclpy.create_node('RGB_detection')

    pub_image = node.create_publisher(Image, output_topic_prefix + 'image', 1)
    pub_boxes = node.create_publisher(String, output_topic_prefix + 'boxes', 1)

    # create a publisher for the output topic for each object class
    for objectClass_id in classesDict:
        objectClass = classesDict[objectClass_id]
        output_topic = pose_topic_prefix + objectClass
        pub = node.create_publisher(PoseStamped, output_topic, 1)
        pub_dict[objectClass_id] = pub

    # queue of 1: drop old frames if inference is slower than the camera
    node.create_subscription(Image, rgb_image_topic, rgb_callback, 1)
    node.create_subscription(CameraInfo, camera_info_topic, camera_info_callback, 1)
    node.get_logger().info(f'Detecting {list(classesDict.values())} on {rgb_image_topic} with {weights}')

    rclpy.spin(node)


if __name__ == '__main__':
    listener()
