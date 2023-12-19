import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  
vel_pub = rospy.Publisher('vel_msg', Twist, queue_size=10)# local model
# rospy.init_node('vel_publisher', anonymous=True)
turn_flag = False

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def move(direction,state_flag,nurse_state):
    global vel_pub
    vel_msg = Twist()
    if state_flag == 'stop':
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.angular.z = 0
        print("I am stopped")
    elif state_flag == 'cone':
        if direction == 'straight':
            vel_msg.linear.x = 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0
            print("I am going straight")
        elif direction == 'left':
            vel_msg.linear.x = 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0.5
            print("I am going left")
            rospy.sleep(2)
        elif direction == 'right':
            vel_msg.linear.x = 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = -0.5
            print("I am going right")
            rospy.sleep(2)
    elif state_flag == 'nurse':
        flag_dir = 0 #前后方向判断
        if nurse_state[1] == 'forward':
            flag_dir = 1
            print("I am going straight")
        elif nurse_state[1] == 'backword':
            flag_dir = -1
            print("I am going backword")
        elif nurse_state[1] == 'stop':
            flag_dir = 0
            print("I am stopped")
        if nurse_state[0] == 'straight':
            vel_msg.linear.x = flag_dir * 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0
            print("I am going straight")
        elif nurse_state[0] == 'left':
            vel_msg.linear.x = flag_dir * 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0.5
            print("I am going left")
            rospy.sleep(2)
        elif nurse_state[0] == 'right':
            vel_msg.linear.x = flag_dir * 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = -0.5
            print("I am going right")
            rospy.sleep(2)

    vel_pub.publish(vel_msg)
    print("I have published a velocity message")

def callback(data):
    print("I have received an image")
    global turn_flag, vel_pub
    direction = 'straight'
    state_flag = 'stop'
    state_nurse = ['stop','stop'];
    # Convert the ROS Image message to OpenCV format
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "rgb8")
    height_orgin = img.shape[0];
    width_orgin = img.shape[1];

    results = model(img)  # inference
    cones = results.pandas().xyxy[0]

    # Confidence
    cones = cones[cones['confidence'] > 0.5]
    print(f"Number of detected cones: {len(cones)}")

    if len(cones) > 1:
        state_flag = 'cone'
    # Get the center of detected cones
        for i in range(len(cones)):
            cones['center_x'] = (cones['xmin'] + cones['xmax']) / 2
            cones['center_y'] = (cones['ymin'] + cones['ymax']) / 2
            cones['height'] = cones['ymax'] - cones['ymin']

        if len(cones) > 0 and len(cones) < 3 and (not turn_flag):
            # Get the closest cone
            closest_cone = cones[cones['height'] == cones['height'].max()]
            print(f"Closest cone: {closest_cone['center_x'].values[0]}, {closest_cone['center_y'].values[0]}")
            # Get the farthest cone
            farthest_cone = cones[cones['height'] == cones['height'].min()]
            print(f"Farthest cone: {farthest_cone['center_x'].values[0]}, {farthest_cone['center_y'].values[0]}")

            if closest_cone['center_x'].values[0] < img.shape[1] / 2:
                print("Turn right")
                direction = 'right'
                turn_flag = True
            else:
                print("Turn left")
                direction = 'left'
                turn_flag = True

        # Draw bounding boxes and confidence
        for i in range(len(cones)):
            cv2.putText(img, f"{cones['confidence'][i]:.2f}", (int(cones['xmin'][i]), int(cones['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            center = (int(cones['center_x'][i]), int(cones['center_y'][i]))
            cv2.circle(img, center, 5, (0, 255, 0), -1)
            cv2.putText(img, f"{center}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (int(cones['xmin'][i]), int(cones['ymin'][i])), (int(cones['xmax'][i]), int(cones['ymax'][i])), (0, 255, 0), 2)
        cv2.imshow("Detection", img)
        cv2.waitKey(1)

    #nurse
    else:
        a = 0.25# 面积阈值权重
        b = 0.4 # 追踪护士距离的阈值
        c = 0.05 # 追踪护士的正负误差
        k = 30 #直行阈值
        area_threshold = a * height_orgin * width_orgin
        area_threshold_track = b * height_orgin * width_orgin
        # 将BGR图像转换为HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 设置颜色的HSV范围
        # 红色在HSV中有两个范围，因为它位于色轮的起点和终点
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # 创建两个掩码以过滤出红色范围内的颜色，并合并这两个掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 使用Canny算法检测边缘
        edges = cv2.Canny(mask, 100, 200)

        # 找到轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
                # 假设最大的轮廓是目标
                contour_max = max(contours, key=cv2.contourArea) #最大轮廓
                area = cv2.contourArea(contour_max) #面积
                if area >= area_threshold:
                    state_flag = 'nurse'
                    print("I have found a nurse")

        if state_flag == 'nurse':
                #中心点
                M = cv2.moments(contour_max)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                #控制小车方向
                if  cX <  width_orgin/2 - k:
                    print("Turn left")
                    state_nurse[0] = 'left'
                elif cX > width_orgin/2 + k:
                    print("Turn right")
                    state_nurse[0] = 'right'
                else:
                    print("Go straight")
                    state_nurse[0] = 'straight'
                #控制小车速度
                if area > area_threshold_track * (1 + c):
                    state_nurse[1] = 'backword'
                elif area < area_threshold_track * (1 - c):
                    state_nurse[1] = 'forward'
                else:
                    state_nurse[1] = 'stop'
                    


                

    # Publish velocity message
    move(direction,state_flag,state_nurse)

    

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()