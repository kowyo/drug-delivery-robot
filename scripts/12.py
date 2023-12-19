


image =  0 #订阅摄像头节点
size = cv.imread(image)
height_orgin = size[0];
width_orgin = size[1];
a = 0.25; # 阈值权重
area_threshold =a * height_orgin * width_orgin
flag_nursestart = 0
# 将BGR图像转换为HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
              flag_nursestart = 1
        
        M = cv2.moments(contour_max)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 控制小车
            control_car(cX, cY, frame.shape[1], frame.shape[0])
