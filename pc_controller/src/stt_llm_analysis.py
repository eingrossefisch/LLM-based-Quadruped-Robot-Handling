import os
import json
import requests
import time
import ast  # 用于安全解析字典字符串
import robot_control_set as robot  # 导入机器人控制模块
import websocket
import threading
import sounddevice as sd
import numpy as np
import uuid
import base64
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def execute_robot_actions(action_dict, port='COM5'):
    """根据动作字典执行机器人动作

    Args:
        action_dict (dict): 动作字典，键为动作名称，值为参数
        port (str): 串口号，默认为'COM5'

    Returns:
        bool: 是否成功执行所有动作
    """
    if not action_dict:
        print("错误: 动作字典为空")
        return False

    # 初始化机器人
    try:
        robot_controller = robot.RobotController(port=port)
        print(f"已连接到机器人，端口: {port}")

        # 执行动作
        for action, value in action_dict.items():
            print(f"执行动作: {action}, 参数: {value}")

            # 如果值不是数字但需要数值，则使用默认值3.0秒
            if action in ["forward", "backward", "left", "right"] and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    print(f"无效的时间值: {value}，使用默认值 3.0 秒")
                    value = 3.0

            if action == "forward":
                # 启动前进运动
                robot_controller.forward()
                # 运行指定时间
                time.sleep(float(value))
                # 停止
                robot_controller.stop()
            elif action == "backward":
                # 启动后退运动
                robot_controller.backward()
                # 运行指定时间
                time.sleep(float(value))
                # 停止
                robot_controller.stop()
            elif action == "left":
                # 启动左转
                robot_controller.left()
                # 运行指定时间
                time.sleep(float(value))
                # 停止
                robot_controller.stop()
            elif action == "right":
                # 启动右转
                robot_controller.right()
                # 运行指定时间
                time.sleep(float(value))
                # 停止
                robot_controller.stop()
            elif action == "takeup" and value:
                # 使用takeup方法自动寻找并抓取红色物体
                robot_controller.takeup()
            elif action == "putdown" and value:
                # 使用putdown方法自动寻找并放下物体到绿色区域
                robot_controller.putdown()
            elif action == "grasp" and value:
                # 执行抓取动作
                robot_controller.grasp()
            elif action == "release" and value:
                # 执行释放动作
                robot_controller.release()
            else:
                print(f"未知动作或参数无效: {action}: {value}")

            # 执行动作之间的短暂延迟
            time.sleep(0.5)

        print("所有动作执行完成")
        return True

    except Exception as e:
        print(f"执行动作时出错: {e}")
        return False


# 解析字符串为字典的函数
def parse_action_dict(text):
    """将字符串解析为Python字典"""
    try:
        # 处理JavaScript风格的true/false
        text = text.replace('true', 'True').replace('false', 'False')

        # 使用ast.literal_eval安全地解析字符串
        action_dict = ast.literal_eval(text)
        if isinstance(action_dict, dict):
            return action_dict
        else:
            print(f"解析结果不是字典: {text}")
            return None
    except (ValueError, SyntaxError) as e:
        print(f"解析错误: {e}")
        print(f"原始文本: {text}")
        return None


# 发送到大模型API的函数
def chat_with_llm(user_input):
    # API 设置
    api_key = os.getenv("LLM_API_KEY")  # 从环境变量读取
    if not api_key:
        print("错误: 未设置LLM_API_KEY环境变量")
        return None

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 系统提示词
    system_prompt = """你是一个先进的四足机器人，能够执行各种动作指令。
你的任务是将用户的指令转换为简单的动作字典。

你可以执行的基本动作包括：
- "forward": 向前移动，值为移动持续时间（秒），建议至少3秒
- "backward": 向后移动，值为移动持续时间（秒），建议至少3秒
- "left": 向左转，值为转动持续时间（秒），每90度大概1.5秒
- "right": 向右转，值为转动持续时间（秒），每90度大概1.5秒
- "takeup": 自动寻找红色物体并夹取，值为True
- "putdown": 自动寻找绿色区域并释放物体，值为True
- "grasp": 执行夹取动作，值为True
- "release": 执行释放动作，值为True

请直接返回一个Python字典格式的字符串，键为动作名称，值为参数。
例如：{"forward": 3.0, "left": 3.0} 或 {"takeup": True, "forward": 5.0}

不要包含任何其他文本或解释，只返回可以被Python的ast.literal_eval()函数解析的字典字符串。"""

    # 构建请求数据
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    # 发送请求
    try:
        print("发送请求到大模型API...")
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # 获取模型回复
        response_text = result["choices"][0]["message"]["content"]
        print("\n大模型回复:")
        print(response_text)

        # 尝试解析为字典
        action_dict = parse_action_dict(response_text)
        if action_dict:
            print("\n解析后的动作字典:")
            print(json.dumps(action_dict, ensure_ascii=False, indent=2))
            return action_dict
        return None

    except requests.exceptions.RequestException as e:
        print(f"\n发生错误: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"错误详情: {response.text}")
        return None


def main():
    """主程序入口，使用百度短语音识别极速版API，一句话识别一句话"""
    print("\n===== 语音控制机器人系统 =====")
    print("使用百度短语音识别极速版API（说一句识别一句）")

    # 百度语音识别API配置 - 从环境变量读取
    API_KEY = os.getenv("BAIDU_API_KEY")
    SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")
    APP_ID = os.getenv("BAIDU_APP_ID")
    robot_port = os.getenv("ROBOT_PORT", "COM5")  # 默认COM5

    # 检查必要的环境变量
    if not all([API_KEY, SECRET_KEY, APP_ID]):
        print("错误: 请设置以下环境变量:")
        print("- BAIDU_API_KEY")
        print("- BAIDU_SECRET_KEY")
        print("- BAIDU_APP_ID")
        print("请参考.env.example文件创建.env文件")
        return

    # 转换APP_ID为整数
    try:
        APP_ID = int(APP_ID)
    except ValueError:
        print("错误: BAIDU_APP_ID必须是数字")
        return

    # 获取百度token
    def get_baidu_token(api_key, secret_key):
        url = f"https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key
        }
        res = requests.post(url, params=params)
        return res.json()["access_token"]

    print("获取百度语音识别API Token...")
    token = get_baidu_token(API_KEY, SECRET_KEY)
    print("Token获取成功!")

    def record_until_silence(samplerate=16000, channels=1, dtype='int16', blocksize=3200, silence_sec=1.0,
                             threshold=200):
        """录音直到静音1秒，返回完整PCM数据"""
        print("请开始说话...")
        audio_buffer = []
        has_voice = False
        last_voice_time = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal last_voice_time, has_voice
            audio_buffer.append(bytes(indata))
            # 判断音量
            audio_np = np.frombuffer(indata, dtype=dtype)
            volume = np.abs(audio_np).mean()
            print(f"当前音量: {volume:.2f}", end="\r")
            if volume > threshold:
                if not has_voice:
                    print("\n检测到声音，正在录音...")
                    has_voice = True
                last_voice_time = time.time()

        with sd.RawInputStream(samplerate=samplerate, channels=channels, dtype=dtype, blocksize=blocksize,
                               callback=callback):
            while True:
                time.sleep(0.1)
                current_silence = time.time() - last_voice_time
                if has_voice and current_silence > silence_sec:
                    break
                # 如果超过10秒还没有声音，自动退出
                if not has_voice and time.time() - last_voice_time > 10:
                    print("\n未检测到声音，请尝试调整麦克风或提高音量")
                    break

        if has_voice:
            print("\n检测到静音，录音结束。")
        return b''.join(audio_buffer)

    def recognize_speech(audio_data):
        """使用百度短语音识别极速版API识别音频"""
        url = "https://vop.baidu.com/pro_api"
        headers = {
            'Content-Type': 'application/json'
        }

        # 将音频数据编码为base64
        speech_data = base64.b64encode(audio_data).decode('utf-8')

        data = {
            "format": "pcm",
            "rate": 16000,
            "channel": 1,
            "cuid": str(APP_ID),
            "token": token,
            "dev_pid": 80001,  # 极速版普通话模型
            "speech": speech_data,
            "len": len(audio_data)
        }

        print(f"发送识别请求，音频长度: {len(audio_data)} 字节")
        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        print(f"接收识别结果: {json.dumps(result, ensure_ascii=False)}")

        if result.get("err_no") == 0 and "result" in result:
            return result["result"][0]
        else:
            print(f"识别错误: {result.get('err_msg', '未知错误')}, 错误码: {result.get('err_no', '未知')}")
            return None

    def process_command():
        """录音、识别、执行命令的循环"""
        while True:
            try:
                # 录音直到静音
                pcm_data = record_until_silence()
                if len(pcm_data) == 0:
                    print("未获取到音频数据，请检查麦克风")
                    time.sleep(1)
                    continue

                print(f"获取到音频数据，长度: {len(pcm_data)} 字节")

                # 发送到百度API识别
                result_text = recognize_speech(pcm_data)

                if result_text:
                    print("\n识别结果：" + result_text)
                    print("-" * 50)
                    print(result_text)
                    print("-" * 50)

                    # 发送到大模型处理
                    print("将语音内容发送到大模型处理中...")
                    action_dict = chat_with_llm(result_text)
                    if action_dict:
                        print(f"\n开始执行机器人动作，端口: {robot_port}")
                        execute_robot_actions(action_dict, robot_port)
                    else:
                        print("大模型未返回有效动作")

                # 等待一段时间再继续下一轮
                time.sleep(1)

            except Exception as e:
                print(f"处理过程中出错: {e}")
                time.sleep(1)

    # 开始主循环
    print("开始录音识别循环，按Ctrl+C退出...")
    try:
        process_command()
    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，程序退出")


if __name__ == "__main__":
    main()