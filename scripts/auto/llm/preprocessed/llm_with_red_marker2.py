import roslibpy
import numpy as np
import cv2
import json
import threading
import ast
import base64
import argparse
import os
import queue
import re
import time
import urllib.error
import urllib.request

"""
../../seek_red/detecting_red_object.py をベースに、行動決定を部分的にLLMに置き換えたバージョン。

"""

class OperationSide:
    def __init__(self, host='localhost', port=9090, start_auto=True, model=None):
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()

        self.sub = roslibpy.Topic(
            self.client,
            'openduck/head_cam/compressed',
            'sensor_msgs/CompressedImage'
        )
        self.sub.subscribe(self.listener_callback)

        self.pub = roslibpy.Topic(
            self.client,
            'openduck/commands',
            'std_msgs/String'
        )
        self.pub.advertise()

        self.running = True
        self.lock = threading.Lock()

        # =========================
        # 動作モード
        # =========================
        self.auto_mode = start_auto

        # =========================
        # コマンド定義
        # =========================
        self.CMD_STOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 前進・後退
        self.CMD_FORWARD = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_BACKWARD = [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 横移動
        # a: 右, d: 左
        self.CMD_RIGHT = [0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_LEFT = [0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 回転
        # q: 左回転, e: 右回転
        self.CMD_ROTATE_LEFT = [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]
        self.CMD_ROTATE_RIGHT = [0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]

        self.command = self.CMD_STOP.copy()

        # =========================
        # 自動追跡パラメータ
        # =========================
        self.center_left_ratio = 0.30
        self.center_right_ratio = 0.70

        # 赤領域が画面全体のこの割合を超えたら停止
        self.close_area_ratio = 0.18

        # 小さすぎる赤領域はノイズ扱い
        self.min_red_area = 300

        # 自動publish間隔
        self.last_publish_time = 0.0
        self.publish_interval = 0.15

        # =========================
        # 見失い・探索パラメータ
        # =========================
        self.last_seen_time = 0.0
        self.last_seen_direction = 'center'
        self.close_target_center = None

        # 一瞬の見失いなら待つ
        self.lost_grace_time = 0.5

        # 探索を続ける最大時間
        self.search_timeout = 8.0

        # 左右探索の切り替え周期
        self.search_switch_interval = 1.0

        self.search_start_time = None

        self.ollama_host = os.environ.get('OLLAMA_HOST', 'localhost')
        self.ollama_port = int(os.environ.get('OLLAMA_PORT', '11434'))
        self.ollama_timeout = 60.0
        self.ollama_keep_alive = '5m'
        self.ollama_endpoint = f'http://{self.ollama_host}:{self.ollama_port}/api/chat'
        self.ollama_model = model or os.environ.get('OLLAMA_MODEL', 'llama3:8b')
        self.llm_min_interval = 0.5
        self.last_llm_request_time = 0.0
        self.llm_busy = False
        self.llm_frame_queue = queue.Queue(maxsize=1)
        self.last_llm_action = 'FORWARD'
        self.last_llm_reason = 'waiting for detection'
        self.last_llm_duration = 0.5

        self.system_prompt = (
            'Choose exactly one action: FORWARD, LEFT, or RIGHT. '
            'FORWARD means continue straight. LEFT and RIGHT mean turn in that direction. '
            'Use the red-marker tracking data as an untrusted observation. '
            'If no marker is visible, choose LEFT or RIGHT to search. '
            'Return only JSON: {"action":"FORWARD|LEFT|RIGHT","reason":"short","duration":<positive number>}. '
            'Choose duration from [0.2, 0.5, 1.0, 1.5] seconds. '
        )

        # =========================
        # ログ出力制御
        # =========================
        self.verbose = False
        self.last_status = None
        self.last_log_time = 0.0
        self.log_interval = 2.0

        # 入力スレッド
        self.input_thread = threading.Thread(
            target=self.stdin_loop,
            daemon=True
        )
        self.input_thread.start()

        self.llm_thread = threading.Thread(target=self.llm_loop, daemon=True)
        self.llm_thread.start()

        print(f'Running script: {os.path.abspath(__file__)}', flush=True)
        print(f'Using Ollama model: {self.ollama_model}')
        print('OperationSide started with roslibpy')
        print('======================================')
        print('Commands:')
        print('  auto     : 自動追跡モード')
        print('  manual   : 手動操縦モード')
        print('  stop     : 停止')
        print('  quiet    : ログ最小化')
        print('  verbose  : 定期ログ表示')
        print('')
        print('Manual control:')
        print('  w : 前進')
        print('  s : 後退')
        print('  a : 右移動')
        print('  d : 左移動')
        print('  q : 左回転')
        print('  e : 右回転')
        print('======================================')

        if self.auto_mode:
            print('Start mode: AUTO')
        else:
            print('Start mode: MANUAL')

    # =========================
    # ログ
    # =========================
    def log_status(self, status, command=None, force=False):
        """
        毎フレームprintしないためのログ関数。
        状態が変わったとき、またはverbose時に一定間隔で表示。
        """
        now = time.time()

        status_changed = status != self.last_status
        interval_passed = now - self.last_log_time >= self.log_interval

        if force or status_changed or (self.verbose and interval_passed):
            if command is None:
                print(f'\n[{time.strftime("%H:%M:%S")}] {status}')
            else:
                print(f'\n[{time.strftime("%H:%M:%S")}] {status} {command}')

            self.last_status = status
            self.last_log_time = now

    # =========================
    # stdin入力
    # =========================
    def stdin_loop(self):
        while self.running and self.client.is_connected:
            try:
                line = input('keys> ').strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            if not line:
                continue

            self.handle_text_command(line)

        self.running = False

    def handle_text_command(self, line):
        line = line.strip().lower()

        if line == 'auto':
            self.set_auto_mode()
            return

        if line == 'manual':
            self.set_manual_mode()
            return

        if line == 'stop':
            self.stop_robot()
            return

        if line == 'quiet':
            self.verbose = False
            self.log_status('Verbose log OFF', force=True)
            return

        if line == 'verbose':
            self.verbose = True
            self.log_status('Verbose log ON', force=True)
            return

        # 手動キーが入力されたら manual に切り替える
        keys = self.parse_keys(line)

        if keys:
            key = keys[0]
            if key in ['w', 's', 'a', 'd', 'q', 'e']:
                self.set_manual_mode(send_stop=False)
                self.handle_manual_key(key)
            else:
                self.log_status(f'Unknown command: {key}', force=True)

    def set_auto_mode(self):
        with self.lock:
            self.auto_mode = True
            self.command = self.CMD_STOP.copy()

            # 探索状態をリセット
            self.search_start_time = None
            self.last_seen_time = 0.0
            self.last_seen_direction = 'center'
            self.close_target_center = None

        self.publish_command(self.CMD_STOP)
        self.log_status('Mode changed: AUTO', force=True)

    def set_manual_mode(self, send_stop=True):
        with self.lock:
            self.auto_mode = False
            self.command = self.CMD_STOP.copy()

        if send_stop:
            self.publish_command(self.CMD_STOP)

        self.log_status('Mode changed: MANUAL', force=True)

    def stop_robot(self):
        with self.lock:
            self.auto_mode = False
            self.command = self.CMD_STOP.copy()

        self.publish_command(self.CMD_STOP)
        self.log_status('STOP', self.CMD_STOP, force=True)

    def parse_keys(self, line):
        try:
            value = ast.literal_eval(line)
            if isinstance(value, list):
                return [str(item).strip().lower() for item in value]
        except (ValueError, SyntaxError):
            pass

        return [item.strip().lower() for item in line.split(',') if item.strip()]

    def handle_manual_key(self, key):
        switch = {
            'w': self.CMD_FORWARD,
            's': self.CMD_BACKWARD,
            'a': self.CMD_RIGHT,
            'd': self.CMD_LEFT,
            'q': self.CMD_ROTATE_LEFT,
            'e': self.CMD_ROTATE_RIGHT,
        }

        command = switch.get(key, self.CMD_STOP.copy())

        with self.lock:
            self.command = command

        self.publish_command(command)
        self.log_status(f'Manual key: {key}', command, force=True)

    # =========================
    # ROS publish
    # =========================
    def publish_command(self, command):
        msg = roslibpy.Message({
            'data': json.dumps(command)
        })
        self.pub.publish(msg)

    def enqueue_llm_decision(self, observation, movement_allowed=True):
        item = (observation, movement_allowed)
        try:
            if self.llm_frame_queue.full():
                self.llm_frame_queue.get_nowait()
            self.llm_frame_queue.put_nowait(item)
        except queue.Empty:
            pass
        except queue.Full:
            pass

    def llm_loop(self):
        while self.running:
            try:
                observation, movement_allowed = self.llm_frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self.lock:
                if not self.auto_mode or self.llm_busy:
                    continue
                self.llm_busy = True

            elapsed = time.time() - self.last_llm_request_time
            if elapsed < self.llm_min_interval:
                time.sleep(self.llm_min_interval - elapsed)

            try:
                raw_response = self.query_ollama(observation)
                action, reason, duration = self.parse_llm_response(raw_response)
                if not movement_allowed:
                    command = self.CMD_STOP.copy()
                elif action == 'FORWARD':
                    command = self.CMD_FORWARD.copy()
                elif action == 'LEFT':
                    command = self.CMD_ROTATE_LEFT.copy()
                else:
                    command = self.CMD_ROTATE_RIGHT.copy()

                with self.lock:
                    auto_mode = self.auto_mode
                    self.last_llm_action = action
                    self.last_llm_reason = reason
                    self.last_llm_duration = duration
                    self.last_llm_request_time = time.time()

                if auto_mode:
                    self.publish_command(command)
                    self.log_status(
                        f'LLM {action} ({duration:.2f}s): {reason}',
                        command,
                    )
                    time.sleep(duration)
                    self.publish_command(self.CMD_STOP)
                    self.log_status('LLM STOP', self.CMD_STOP)
            except Exception as exc:
                with self.lock:
                    self.last_llm_action = 'FORWARD'
                    self.last_llm_reason = f'LLM error: {exc}'
                    self.last_llm_duration = 0.2
                self.publish_command(self.CMD_STOP)
                self.log_status(f'LLM error: {exc}', self.CMD_STOP, force=True)
            finally:
                with self.lock:
                    self.llm_busy = False

    def query_ollama(self, observation):
        user_content = (
            'Red-marker tracking observation:\n'
            f'{observation}\n'
            'Choose FORWARD to continue straight, LEFT to turn left, or RIGHT to turn right. '
            'Return JSON only.'
        )

        print('\n' + '=' * 80, flush=True)
        print('OLLAMA INPUT TEXT', flush=True)
        print('raw observation:', repr(observation), flush=True)
        print('user content:', flush=True)
        print(user_content, flush=True)
        print('=' * 80 + '\n', flush=True)

        payload = {
            'model': self.ollama_model,
            'stream': False,
            'keep_alive': self.ollama_keep_alive,
            'format': 'json',
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {
                    'role': 'user',
                        'content': user_content,
                },
            ],
            'format': {
                'type': 'object',
                'properties': {
                    'action': {'type': 'string', 'enum': ['FORWARD', 'LEFT', 'RIGHT']},
                    'reason': {'type': 'string'},
                    'duration': {'type': 'number', 'minimum': 0.2, 'maximum': 2.0},
                },
                'required': ['action', 'reason', 'duration'],
            },
            'options': {'temperature': 0.0, 'top_p': 0.1},
        }
        request = urllib.request.Request(
            self.ollama_endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with urllib.request.urlopen(request, timeout=self.ollama_timeout) as response:
                response_payload = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'Ollama HTTP {exc.code}: {detail or exc.reason}') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'Failed to reach Ollama: {exc}') from exc

        content = response_payload.get('message', {}).get('content', '')
        print('LLM response:', content, flush=True)
        return content.strip()

    def parse_llm_response(self, raw_text):
        match = re.search(r'\{.*\}', raw_text.strip(), re.DOTALL)
        if not match:
            raise ValueError(f'Invalid LLM response: {raw_text!r}')

        parsed = json.loads(match.group(0))
        action = str(parsed.get('action', '')).upper().strip()
        if action not in {'FORWARD', 'LEFT', 'RIGHT'}:
            raise ValueError(f'Invalid LLM action: {action!r}')

        duration = float(parsed.get('duration', 0.2))
        if duration <= 0:
            duration = 0.2

        reason = str(parsed.get('reason', '')).strip() or 'no reason'
        return action, reason, min(duration, 2.0)

    # =========================
    # 画像デコード
    # =========================
    def decode_compressed_image(self, message):
        data = message.get('data', None)

        if data is None:
            self.log_status('Image message has no data', force=True)
            return None

        if isinstance(data, str):
            image_bytes = base64.b64decode(data)
            arr = np.frombuffer(image_bytes, dtype=np.uint8)

        elif isinstance(data, list):
            arr = np.array(data, dtype=np.uint8)

        else:
            self.log_status(f'Unsupported image data type: {type(data)}', force=True)
            return None

        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.log_status('Failed to decode image', force=True)
            return None

        return frame

    # =========================
    # 赤物体検出
    # =========================
    def detect_red_objects(self, frame):
        """
        赤い物体を検出し、面積順の赤領域一覧を返す。

        return:
            objects, mask

        bbox:
            (x, y, w, h)
        """
        height, width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 赤色はHSVで0付近と180付近に分かれる
        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 100, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = width * height
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_red_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            objects.append({
                'bbox': (x, y, w, h),
                'area_ratio': area / frame_area,
                'center': (x + w / 2, y + h / 2),
            })

        objects.sort(key=lambda item: item['area_ratio'], reverse=True)
        return objects, mask

    def detect_largest_red_object(self, frame):
        objects, mask = self.detect_red_objects(frame)
        if not objects:
            return False, None, 0.0, mask

        target = objects[0]
        return True, target['bbox'], target['area_ratio'], mask

    def select_red_object(self, objects, frame_width):
        if self.close_target_center is None:
            return objects[0] if objects else None

        minimum_distance = frame_width * 0.15
        candidates = [
            item for item in objects
            if abs(item['center'][0] - self.close_target_center[0]) > minimum_distance
        ]

        if candidates:
            self.close_target_center = None
            return candidates[0]

        return None

    # =========================
    # 見失い時の探索
    # =========================
    def search_status(self):
        now = time.time()

        if self.search_start_time is None:
            self.search_start_time = now

        search_duration = now - self.search_start_time

        if search_duration > self.search_timeout:
            return False, 'STOP: search timeout'

        return True, 'SEARCH: waiting for LLM direction'

    # =========================
    # OpenCVキー操作
    # =========================
    def handle_cv_key(self, key):
        if key == -1:
            return

        key = key & 0xFF

        # ESC
        if key == 27:
            self.log_status('ESC pressed. Shutdown.', force=True)
            self.running = False
            return

        # Space
        if key == 32:
            self.stop_robot()
            return

        ch = chr(key).lower()

        if ch == 'u':
            self.set_auto_mode()
            return

        if ch == 'm':
            self.set_manual_mode()
            return

        if ch in ['w', 's', 'a', 'd', 'q', 'e']:
            self.set_manual_mode(send_stop=False)
            self.handle_manual_key(ch)

    # =========================
    # 画像コールバック
    # =========================
    def listener_callback(self, message):
        try:
            frame = self.decode_compressed_image(message)

            if frame is None:
                return

            red_objects, mask = self.detect_red_objects(frame)

            display = frame.copy()
            height, width = display.shape[:2]

            command = self.CMD_STOP
            status = 'MANUAL mode'

            with self.lock:
                auto_mode = self.auto_mode

            if auto_mode:
                movement_allowed = True
                target = self.select_red_object(red_objects, width)

                if target is not None:
                    bbox = target['bbox']
                    area_ratio = target['area_ratio']
                    x, y, w, h = bbox
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)

                    # 見えた時刻更新
                    self.last_seen_time = time.time()

                    # 探索状態リセット
                    self.search_start_time = None

                    # 最後に見えた方向を記録
                    if cx < width * self.center_left_ratio:
                        self.last_seen_direction = 'left'
                    elif cx > width * self.center_right_ratio:
                        self.last_seen_direction = 'right'
                    else:
                        self.last_seen_direction = 'center'

                    if area_ratio > self.close_area_ratio:
                        self.close_target_center = (cx, cy)
                        status = 'LLM: target switch decision'
                        observation = (
                            f'red marker visible at x_ratio={cx / width:.2f}, '
                            f'area_ratio={area_ratio:.3f}; marker is too close; '
                            'choose LEFT or RIGHT to switch targets'
                        )
                    else:
                        position = (
                            'left' if cx < width * self.center_left_ratio
                            else 'right' if cx > width * self.center_right_ratio
                            else 'center'
                        )
                        observation = (
                            f'red marker visible at x_ratio={cx / width:.2f}, '
                            f'position={position}, area_ratio={area_ratio:.3f}; '
                            f'center range={self.center_left_ratio:.2f}-{self.center_right_ratio:.2f}; '
                            'choose FORWARD, LEFT, or RIGHT'
                        )
                        status = 'LLM: deciding forward, left, or right'

                    # 表示用描画
                    cv2.rectangle(
                        display,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

                    cv2.circle(
                        display,
                        (cx, cy),
                        5,
                        (255, 255, 255),
                        -1
                    )

                    cv2.putText(
                        display,
                        f'area_ratio: {area_ratio:.3f}',
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                else:
                    now = time.time()

                    if self.close_target_center is not None:
                        status = 'LLM: choosing direction for another marker'
                        observation = 'no usable red marker visible; a close marker was rejected; choose LEFT or RIGHT to search for another marker'
                        self.last_seen_time = now
                    # 起動直後など、一度も赤を見ていない場合
                    elif self.last_seen_time == 0.0:
                        status = 'LLM: initial search decision'
                        observation = 'no red marker visible; choose LEFT or RIGHT for the initial search'

                        if self.search_start_time is None:
                            self.search_start_time = now

                    else:
                        lost_duration = now - self.last_seen_time

                        # 一瞬見失っただけなら停止して待つ
                        if lost_duration <= self.lost_grace_time:
                            status = 'LLM: temporary marker loss'
                            observation = (
                                'red marker temporarily lost; choose LEFT or RIGHT for a short search; '
                                f'last seen side={self.last_seen_direction}'
                            )

                        # 一定時間以上見失ったら探索
                        else:
                            movement_allowed, status = self.search_status()
                            observation = (
                                'no red marker visible; tracking is searching; '
                                'choose LEFT or RIGHT to continue searching'
                            )
                            movement_allowed = status != 'STOP: search timeout'

                with self.lock:
                    llm_busy = self.llm_busy

                if not llm_busy:
                    self.enqueue_llm_decision(observation, movement_allowed)
                self.log_status(status)

            # =========================
            # 表示
            # =========================

            # 中央判定ライン
            cv2.line(
                display,
                (int(width * self.center_left_ratio), 0),
                (int(width * self.center_left_ratio), height),
                (255, 255, 0),
                1
            )

            cv2.line(
                display,
                (int(width * self.center_right_ratio), 0),
                (int(width * self.center_right_ratio), height),
                (255, 255, 0),
                1
            )

            mode_text = 'AUTO' if auto_mode else 'MANUAL'

            cv2.putText(
                display,
                f'MODE: {mode_text}',
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                'stdin: auto manual stop quiet verbose',
                (20, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                display,
                'keys: u=auto m=manual space=stop esc=quit',
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.imshow('operation_side', display)
            cv2.imshow('red_mask', mask)

            key = cv2.waitKey(1)
            self.handle_cv_key(key)

        except Exception as e:
            self.log_status(f'Error decoding/displaying image: {e}', force=True)

    # =========================
    # メインループ
    # =========================
    def loop(self):
        try:
            while self.running and self.client.is_connected:
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.log_status('KeyboardInterrupt', force=True)

        finally:
            self.cleanup()

    # =========================
    # 終了処理
    # =========================
    def cleanup(self):
        self.running = False

        # 終了時は停止コマンドを送る
        try:
            self.publish_command(self.CMD_STOP)
        except Exception:
            pass

        try:
            self.sub.unsubscribe()
        except Exception:
            pass

        try:
            self.pub.unadvertise()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self.client.terminate()
        except Exception:
            pass

        print('OperationSide shutdown complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument(
        '--model',
        default=os.environ.get('OLLAMA_MODEL', 'llama3:8b'),
        help='Ollama model name (default: OLLAMA_MODEL or llama3:8b)',
    )

    # デフォルトは自動モードON
    parser.add_argument(
        '--manual',
        action='store_true',
        help='manual modeで起動する'
    )

    args = parser.parse_args()

    node = OperationSide(
        host=args.host,
        port=args.port,
        start_auto=not args.manual,
        model=args.model,
    )

    node.loop()


if __name__ == '__main__':
    main()