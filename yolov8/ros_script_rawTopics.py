from ultralytics import YOLO
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, Float64
import numpy as np
import message_filters
import os

# RGB-only detection on a raw (non compressed) equirectangular (360°) image stream, triggered by /start_detection_command
# No depth camera: the shelf position is recovered from its known height, assuming it stands upright
# (parallel to the camera's vertical axis). The orientation cannot be observed from a bounding box:
# the published pose has identity orientation, expressed in the camera optical frame (x right, y down, z forward)

classesDict = dict()
#classesDict[0] = 'CupNoodles'
#classesDict[1] = 'CurryCup'
#classesDict[2] = 'SeaFoodCup'
#classesDict[0] = 'BoxNoLid'
# classesDict[0] = 'BoxLid'
# classesDict[1] = 'BoxNoLid'
classesDict[0] = 'shelf'

rgb_image_topic = '/webrtc/image_raw'
weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', 'shelf', 'best.pt')
confidence_threshold = 0.3
shelf_height = 1.5 #meters
border_margin = 2 #pixels, a box this close to the top/bottom of the image is considered truncated

output_topic_prefix = '/object_pose_initialization/'

bridge = CvBridge()
model = YOLO(weights)
pub_dict = dict()
pub_detection_time = None
node = None
pub_end = None

def sync_callback(rgb_msg, command_msg):
    try:
        rgb_data = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
    except CvBridgeError as e:
        node.get_logger().error(str(e))
        return
    process_image(rgb_data, rgb_msg.header)

def pixel_to_angles(u, v, img_width, img_height):
    #equirectangular projection: u is linear in the azimuth, v is linear in the elevation
    #returns (azimuth, elevation) in radians, azimuth positive to the right, elevation positive downwards
    alpha_u = img_width * 0.5 / np.pi
    alpha_v = img_height * 0.5 / (np.pi * 0.5)
    u_0 = img_width * 0.5
    v_0 = img_height * 0.5
    return (u - u_0) / alpha_u, (v - v_0) / alpha_v

def estimate_position(x1, y1, x2, y2, img_width, img_height):
    #the top and bottom edges of the box are the top and bottom of the shelf, seen at the same azimuth
    #at a horizontal distance D: tan(elevation_bottom) - tan(elevation_top) = shelf_height / D
    #returns (X, Y, Z) of the shelf center in meters, in the camera optical frame
    u = (x1 + x2) / 2
    azimuth, elevation_top = pixel_to_angles(u, y1, img_width, img_height)
    _, elevation_bottom = pixel_to_angles(u, y2, img_width, img_height)
    #the tangent is undefined at the poles
    if not (-np.pi / 2 < elevation_top < elevation_bottom < np.pi / 2):
        return None
    t_top = np.tan(elevation_top)
    t_bottom = np.tan(elevation_bottom)
    D = shelf_height / (t_bottom - t_top)
    return D * np.sin(azimuth), D * (t_top + t_bottom) / 2, D * np.cos(azimuth)

def process_image(rgb_data, header):
    img_height, img_width = rgb_data.shape[:2]

    start = node.get_clock().now()
    results = model(rgb_data, conf=confidence_threshold, verbose=False)[0]
    duration = node.get_clock().now() - start
    node.get_logger().info('Inference time (s): %f' % (duration.nanoseconds / 1.0e9))
    msg = Float64()
    msg.data = duration.nanoseconds / 1e9
    pub_detection_time.publish(msg)

    detected_objects_txt = ''
    for cls, conf, xyxy in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
        objectClass_id = int(cls.item())
        objectClass = classesDict.get(objectClass_id)
        #if not found in the dictionary, skip
        if objectClass is None:
            print('Warning: object class not found in the dictionary. Check that the model was trained with the same classes as the dictionary.')
            continue
        detected_objects_txt += objectClass + ' '
        confidence = conf.item()
        x1, y1, x2, y2 = xyxy.cpu().numpy()

        if y1 <= border_margin or y2 >= img_height - 1 - border_margin:
            node.get_logger().warn(f'{objectClass} cut by the top/bottom of the image, pose not published')
            continue
        position = estimate_position(x1, y1, x2, y2, img_width, img_height)
        if position is None:
            node.get_logger().warn(f'Invalid back-projection of the {objectClass} box, pose not published')
            continue
        X, Y, Z = position

        #quat = np.array([0.6129531264305115, 0.6196120977401733, -0.3606013059616089, 0.3321303129196167])
        quat = np.array([0.9537, 0.0, 0.0, 0.3007])

        # publish the result to the output topic
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = float(X) - 0.6
        pose.pose.position.y = float(Y) + shelf_height/2.0 
        pose.pose.position.z = float(Z)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        #pose.pose.orientation.w = 1.0
        pub_dict[objectClass_id].publish(pose)
        node.get_logger().info('Object detected: %s' % objectClass)
        node.get_logger().info('Confidence: %f' % confidence)
        node.get_logger().info(f'Pose: x={X:.2f} m, y={Y:.2f} m, z={Z:.2f} m')

    #send end of detection signal
    msg_str = String()
    msg_str.data = detected_objects_txt
    pub_end.publish(msg_str)


def listener():
    global pub_dict
    global node
    global pub_detection_time
    global pub_end
    # initialize the node
    rclpy.init()
    node = rclpy.create_node('RGB_detection')

    pub_end = node.create_publisher(String, "/end_of_detection", 1)
    pub_detection_time = node.create_publisher(Float64, "/detection_time", 1)

    # create a publisher for the output topic for each object class
    for objectClass_id in classesDict:
        objectClass = classesDict[objectClass_id]
        output_topic = output_topic_prefix + "Shelf" 
        pub = node.create_publisher(PoseStamped, output_topic, 1)
        pub_dict[objectClass_id] = pub

    # run the detection on the image received together with the start command
    sub_rgb = message_filters.Subscriber(node, Image, rgb_image_topic)
    sub_command = message_filters.Subscriber(node, Bool, "/start_detection_command")
    combined_sub = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_command], 1, 0.1, allow_headerless=True)
    combined_sub.registerCallback(sync_callback)

    node.get_logger().info(f'Detecting {list(classesDict.values())} on {rgb_image_topic} with {weights}')

    # spin
    rclpy.spin(node)


if __name__ == '__main__':
    listener()
