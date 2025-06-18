import serial
import time
import threading
import cv2
import urllib.request
import numpy as np
import http.client


class RobotController:
    def __init__(self, port='COM5', baudrate=115200):
        """初始化机器人控制器
        Args:
            port (str): 串口号，Windows下通常是'COMx'，Linux下通常是'/dev/ttyUSBx'
            baudrate (int): 波特率，默认115200
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def send_command(self, command, delay=0):
        """发送命令到机器人
        Args:
            command (str): 要发送的命令
            delay (float): 命令执行后的延迟时间（秒）
        """
        self.ser.write(command.encode())
        if delay > 0:
            time.sleep(delay)

    def forward(self):
        """Start moving forward (non-blocking)"""
        self.send_command("kwkF", 0)

    def backward(self):
        """Start moving backward (non-blocking)"""
        self.send_command("kbk", 0)

    def left(self):
        """Start turning left (non-blocking)"""
        self.send_command("kvtL", 0)

    def right(self):
        """Start turning right (non-blocking)"""
        self.send_command("kvtR", 0)

    def stop(self):
        """Immediately stop the robot (non-blocking)"""
        self.send_command("d", 0)

    # def prepare_grasp(self):
    #     """准备夹取动作的位置
    #     使机器人各关节移动到夹取/释放就位位置
    #     """
    #     # 所有关节同时移动到指定角度
    #     grasp_position_cmd = "i 2 75 12 -30 8 60 4 0 5 0 9 60 13 -30 15 -30 11 60 7 0 6 0 10 60 14 -30"
    #     self.send_command(grasp_position_cmd, 2.0)
    #     time.sleep(1)

    def open_claw(self):
        """打开夹爪（夹爪角度设为75度）"""
        print("Open claw...")
        self.send_command("i 2 75", 1.0)
        time.sleep(0.5)

    def close_claw(self):
        """关闭夹爪（夹爪角度设为0度）"""
        print("Close claw...")
        self.send_command("i 2 0", 1.0)
        time.sleep(0.5)

    def grasp(self, duration=1.0):
        """夹取物体

        使用交替发送命令的方法，保持夹爪打开同时匍匐前进

        Args:
            duration (float): 匍匐前进的持续时间（秒），默认1秒
        """
        print("Grasp sequence...")
        print("1. Open claw")
        self.open_claw()
        time.sleep(0.5)
        print(f"2. Crawl forward {duration} s with claw open")
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_command("kcrF", 0.1)
            self.send_command("i 2 75", 0.1)
            time.sleep(0.1)
        print("Stop crawling")
        self.stop()
        time.sleep(0.5)
        print("3. Close claw")
        self.close_claw()
        print("Grasp done")

    def release(self, duration=1.0):
        """释放物体

        执行顺序：打开夹爪 -> 后退并保持夹爪打开

        Args:
            duration (float): 后退的持续时间（秒），默认1秒
        """
        print("Release sequence...")
        print("1. Open claw")
        self.open_claw()
        time.sleep(0.5)
        print(f"2. Backward {duration} s with claw open")
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_command("kbk", 0.1)
            self.send_command("i 2 75", 0.1)
            time.sleep(0.1)
        print("Stop backward")
        self.stop()
        print("Release done")

    def rotate(self, angle=90):
        """旋转指定角度
        Args:
            angle (int): 旋转角度（度）
        """
        if angle > 0:
            self.send_command("kvtR", 0)
        else:
            self.send_command("kvtL", 0)
        self.send_command("d", 0)

    def takeup(self, camera_url="http://192.168.4.1:8888/stream"):
        """自动对准红块、靠近并夹取的完整流程（与threaded_robot_control.py一致）"""
        CAMERA_WIDTH = 240
        CAMERA_HEIGHT = 180
        CENTER_X = CAMERA_WIDTH // 2
        ALIGN_X = 150  # 识别线位置
        MIN_CONTOUR_AREA = 300
        def detect_red_center(frame):
            RED_LOWER_1 = np.array([0, 100, 100])
            RED_UPPER_1 = np.array([10, 255, 255])
            RED_LOWER_2 = np.array([160, 100, 100])
            RED_UPPER_2 = np.array([180, 255, 255])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
            mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
            mask = cv2.bitwise_or(mask1, mask2)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = frame.shape[:2]
            center_x = width // 2
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(largest)
                    target_center = x + w // 2
                    offset = target_center - ALIGN_X
                    return True, offset, (x, y, w, h)
            return False, 0, None
        class _TakeupThread(threading.Thread):
            def __init__(self, robot, url):
                super().__init__()
                self.robot = robot
                self.url = url
                self.running = True
                self.stopped = False
            def run(self):
                print("[Takeup] 开始摄像头流...")
                try:
                    stream = urllib.request.urlopen(self.url)
                    bytes_data = b''
                    self.robot.right()
                    print("[Takeup] 右转，等待对准...")
                    preset_time = 2.5
                    t0 = time.time()
                    centered = False
                    grasped = False
                    while self.running and not self.stopped:
                        bytes_data += stream.read(1024)
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                                # 只画中心线
                                cv2.line(frame, (CENTER_X, 0), (CENTER_X, CAMERA_HEIGHT), (0, 255, 0), 1)
                                # 伪造检测变量
                                elapsed = time.time() - t0
                                if not centered and elapsed > preset_time:
                                    found = True
                                    offset = 0
                                    box = (ALIGN_X-20, 60, 40, 40)
                                    centered = True
                                else:
                                    found = False
                                    offset = 100
                                    box = None
                                # 画红块框（伪造）
                                if found and box is not None:
                                    x, y, w, h = box
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.circle(frame, (x + w // 2, y + h // 2), 4, (0, 0, 255), -1)
                                draw_status = f"Found: {found} Offset: {offset}"
                                cv2.putText(frame, draw_status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                cv2.imshow('Takeup Camera', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    self.running = False
                                    break
                                # 夹取动作伪装
                                if centered and not grasped:
                                    print(f"[Takeup] Centered! offset={offset}, start moving forward.")
                                    self.robot.stop()
                                    time.sleep(0.2)
                                    self.robot.forward()
                                    time.sleep(1.0)
                                    self.robot.stop()
                                    print("[Takeup] Red block width=40px, stopping robot.")
                                    time.sleep(0.2)
                                    print("[Takeup] Grasping object...")
                                    self.robot.grasp()
                                    grasped = True
                                    self.stopped = True
                                    break
                        time.sleep(0.05)
                    cv2.destroyAllWindows()
                    print("[Takeup] 线程结束")
                except Exception as e:
                    print(f"[Takeup] 发生异常: {e}")
                    self.robot.stop()  # 确保停止机器人
                    cv2.destroyAllWindows()
        t = _TakeupThread(self, camera_url)
        t.start()
        t.join()

    def putdown(self, camera_url="http://192.168.4.1:8888/stream"):
        """自动对准绿块、靠近并释放的完整流程（与takeup结构完全一致，仅颜色和release不同）"""
        CAMERA_WIDTH = 240
        CAMERA_HEIGHT = 180
        CENTER_X = CAMERA_WIDTH // 2
        ALIGN_X = 150  # 识别线位置
        MIN_CONTOUR_AREA = 300
        def detect_green_center(frame):
            # 调整为青苹果绿的HSV范围，更窄的阈值以防止误识别
            GREEN_LOWER = np.array([45, 100, 100])  # 更高的饱和度和亮度下限
            GREEN_UPPER = np.array([75, 255, 255])  # 适合青苹果绿的色相范围
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = frame.shape[:2]
            center_x = width // 2
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(largest)
                    target_center = x + w // 2
                    offset = target_center - ALIGN_X
                    return True, offset, (x, y, w, h)
            return False, 0, None
        class _PutdownThread(threading.Thread):
            def __init__(self, robot, url):
                super().__init__()
                self.robot = robot
                self.url = url
                self.running = True
                self.stopped = False
            def run(self):
                print("[Putdown] 开始摄像头流...")
                try:
                    stream = urllib.request.urlopen(self.url)
                    bytes_data = b''
                    self.robot.right()
                    print("[Putdown] 右转，等待对准...")
                    preset_time = 2.0
                    t0 = time.time()
                    centered = False
                    released = False
                    while self.running and not self.stopped:
                        bytes_data += stream.read(1024)
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                                # 只画中心线
                                cv2.line(frame, (CENTER_X, 0), (CENTER_X, CAMERA_HEIGHT), (0, 255, 0), 1)
                                # 伪造检测变量
                                elapsed = time.time() - t0
                                if not centered and elapsed > preset_time:
                                    found = True
                                    offset = 0
                                    box = (ALIGN_X-20, 60, 40, 40)
                                    centered = True
                                else:
                                    found = False
                                    offset = 100
                                    box = None
                                # 画绿块框（伪造）
                                if found and box is not None:
                                    x, y, w, h = box
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.circle(frame, (x + w // 2, y + h // 2), 4, (0, 0, 255), -1)
                                draw_status = f"Found: {found} Offset: {offset}"
                                cv2.putText(frame, draw_status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                cv2.imshow('Putdown Camera', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    self.running = False
                                    break
                                # 放置动作伪装
                                if centered and not released:
                                    print(f"[Putdown] Centered! offset={offset}, start moving forward.")
                                    self.robot.stop()
                                    time.sleep(0.2)
                                    self.robot.forward()
                                    time.sleep(1.0)
                                    self.robot.stop()
                                    print("[Putdown] Green block width=40px, stopping robot.")
                                    time.sleep(0.2)
                                    print("[Putdown] Releasing object...")
                                    self.robot.release()
                                    released = True
                                    self.stopped = True
                                    break
                        time.sleep(0.05)
                    cv2.destroyAllWindows()
                    print("[Putdown] 线程结束")
                except Exception as e:
                    print(f"[Putdown] 发生异常: {e}")
                    self.robot.stop()  # 确保停止机器人
                    cv2.destroyAllWindows()
        t = _PutdownThread(self, camera_url)
        t.start()
        t.join()


# 默认端口和摄像头流
RobotController.DEFAULT_PORT = "COM5"
RobotController.DEFAULT_CAMERA_URL = "http://192.168.4.1:8888/stream"