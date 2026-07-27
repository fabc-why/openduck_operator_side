"""
起動後にカメラ映像を保存し、stdinで受け取ったJSONコマンドをそのままロボットへ流す。
初回フレームは llm_human_relay_img に保存し、その後もコマンド受信時に現在の画像を保存する。
"""

import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
import roslibpy


class OperationSide:
    def __init__(
        self,
        host='localhost',
        port=9090,
        output_dir=None,
    ):
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

        default_output_dir = Path(__file__).with_name('llm_human_relay_img')
        self.output_dir = Path(
            output_dir or os.environ.get('LLM_HUMAN_RELAY_IMG_DIR') or default_output_dir
        ).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.last_saved_image_path = None
        self.initial_snapshot_saved = False
        self.last_action = 'STOP'
        self.last_reason = 'waiting for the first frame'
        self.last_raw_command = ''

        self.CMD_STOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_FORWARD = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_BACKWARD = [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_RIGHT = [0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_LEFT = [0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.CMD_ROTATE_LEFT = [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]
        self.CMD_ROTATE_RIGHT = [0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]

        self.command = self.CMD_STOP.copy()

        self.verbose = False
        self.last_status = None
        self.last_log_time = 0.0
        self.log_interval = 2.0

        self.input_thread = threading.Thread(target=self.stdin_loop, daemon=True)
        self.input_thread.start()

        print('OperationSide started with roslibpy')
        print('======================================')
        print(f'Image output dir: {self.output_dir}')
        print('Commands:')
        print('  stdin JSON {"action": "...", "reason": "..."}')
        print('  action values        : STOP, FORWARD, BACKWARD, LEFT, RIGHT, ROTATE_LEFT, ROTATE_RIGHT')
        print('')
        print('Manual control:')
        print('  w : forward')
        print('  s : backward')
        print('  a : strafe right')
        print('  d : strafe left')
        print('  q : rotate left')
        print('  e : rotate right')
        print('======================================')
        print('Waiting for camera frames and stdin commands')

    def log_status(self, status, command=None, force=False):
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

    def truncate_text(self, text, limit=90):
        cleaned = ' '.join(str(text).split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1] + '…'

    def stdin_loop(self):
        while self.running and self.client.is_connected:
            try:
                line = input('stdin> ').strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            if not line:
                continue

            self.handle_text_command(line)

        self.running = False

    def handle_text_command(self, line):
        parsed = self.parse_command_input(line)
        if parsed is None:
            return

        action = parsed['action']
        reason = parsed['reason']
        command = self.action_to_command(action)

        with self.lock:
            self.command = command
            self.last_action = action
            self.last_reason = reason
            self.last_raw_command = line

        snapshot_path = self.save_latest_frame_snapshot(prefix=action.lower())
        self.publish_command(command)

        if snapshot_path is not None:
            self.log_status(
                f'Command {action}: {reason} | saved {snapshot_path.name}',
                command,
                force=True,
            )
        else:
            self.log_status(f'Command {action}: {reason}', command, force=True)

    def parse_command_input(self, line):
        text = line.strip()
        if not text:
            return None

        payload = None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            upper = text.upper()
            if upper in {
                'STOP',
                'FORWARD',
                'BACKWARD',
                'LEFT',
                'RIGHT',
                'ROTATE_LEFT',
                'ROTATE_RIGHT',
            }:
                payload = {'action': upper, 'reason': 'plain action input'}
            elif text.lower() in {'w', 's', 'a', 'd', 'q', 'e'}:
                payload = {
                    'action': {
                        'w': 'FORWARD',
                        's': 'BACKWARD',
                        'a': 'RIGHT',
                        'd': 'LEFT',
                        'q': 'ROTATE_LEFT',
                        'e': 'ROTATE_RIGHT',
                    }[text.lower()],
                    'reason': 'keyboard shortcut input',
                }
            else:
                self.log_status('stdin input must be JSON like {"action": "STOP", "reason": "..."}', force=True)
                return None

        if not isinstance(payload, dict):
            self.log_status('stdin JSON must be an object with action and reason', force=True)
            return None

        action = str(payload.get('action', '')).upper().strip()
        reason = str(payload.get('reason', '')).strip() or 'no reason provided'

        allowed_actions = {
            'STOP',
            'FORWARD',
            'BACKWARD',
            'LEFT',
            'RIGHT',
            'ROTATE_LEFT',
            'ROTATE_RIGHT',
        }

        if action not in allowed_actions:
            self.log_status(f'Unknown action: {action}', force=True)
            return None

        return {'action': action, 'reason': reason}

    def save_frame_snapshot(self, frame, prefix='frame'):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        millis = int((time.time() % 1) * 1000)
        path = self.output_dir / f'{prefix}_{timestamp}_{millis:03d}.png'

        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f'Failed to save image to {path}')

        self.last_saved_image_path = path
        return path

    def save_latest_frame_snapshot(self, prefix='frame'):
        with self.lock:
            if self.latest_frame is None:
                self.log_status('No camera frame available yet', force=True)
                return None

            frame = self.latest_frame.copy()

        return self.save_frame_snapshot(frame, prefix=prefix)

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
            self.last_action = key.upper()
            self.last_reason = 'keyboard command'
            self.last_raw_command = key

        self.save_latest_frame_snapshot(prefix=key.lower())
        self.publish_command(command)
        self.log_status(f'Manual key: {key}', command, force=True)

    def publish_command(self, command):
        msg = roslibpy.Message({'data': json.dumps(command)})
        self.pub.publish(msg)

    def decode_compressed_image(self, message):
        data = message.get('data', None)

        if data is None:
            self.log_status('Image message has no data', force=True)
            return None, None

        if isinstance(data, str):
            image_bytes = base64.b64decode(data)
        elif isinstance(data, list):
            image_bytes = bytes(data)
        else:
            self.log_status(f'Unsupported image data type: {type(data)}', force=True)
            return None, None

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.log_status('Failed to decode image', force=True)
            return None

        return frame

    def action_to_command(self, action):
        mapping = {
            'STOP': self.CMD_STOP,
            'FORWARD': self.CMD_FORWARD,
            'BACKWARD': self.CMD_BACKWARD,
            'LEFT': self.CMD_LEFT,
            'RIGHT': self.CMD_RIGHT,
            'ROTATE_LEFT': self.CMD_ROTATE_LEFT,
            'ROTATE_RIGHT': self.CMD_ROTATE_RIGHT,
        }

        command = mapping.get(action, self.CMD_STOP)
        return command.copy()

    def handle_cv_key(self, key):
        if key == -1:
            return

        key = key & 0xFF

        if key == 27:
            self.log_status('ESC pressed. Shutdown.', force=True)
            self.running = False
            return

        if key == 32:
            self.stop_robot()
            return

        ch = chr(key).lower()

        if ch in ['w', 's', 'a', 'd', 'q', 'e']:
            self.handle_manual_key(ch)

    def stop_robot(self):
        with self.lock:
            self.command = self.CMD_STOP.copy()
            self.last_action = 'STOP'
            self.last_reason = 'operator stop'
            self.last_raw_command = 'stop'

        self.save_latest_frame_snapshot(prefix='stop')
        self.publish_command(self.CMD_STOP)
        self.log_status('STOP', self.CMD_STOP, force=True)

    def listener_callback(self, message):
        try:
            frame = self.decode_compressed_image(message)

            if frame is None:
                return

            display = frame.copy()
            height, width = display.shape[:2]

            with self.lock:
                self.latest_frame = frame
                self.latest_frame_time = time.time()
                action = self.last_action
                reason = self.last_reason
                saved_path = self.last_saved_image_path
                initial_snapshot_saved = self.initial_snapshot_saved

            if not initial_snapshot_saved:
                snapshot_path = self.save_frame_snapshot(frame, prefix='startup')
                with self.lock:
                    self.initial_snapshot_saved = True
                    saved_path = snapshot_path

            status = f'LAST: {action} | {reason}'

            cv2.putText(
                display,
                'MODE: RELAY',
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                self.truncate_text(status, 90),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f'SAVED: {self.truncate_text(saved_path.name if saved_path else "none", 70)}',
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display,
                'stdin: JSON {"action": "STOP", "reason": "..."}',
                (20, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display,
                'keys: w/s/a/d/q/e=move space=stop esc=quit',
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow('operation_side', display)

            key = cv2.waitKey(1)
            self.handle_cv_key(key)

        except Exception as exc:
            self.log_status(f'Error decoding/displaying image: {exc}', force=True)

    def loop(self):
        try:
            while self.running and self.client.is_connected:
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.log_status('KeyboardInterrupt', force=True)

        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False

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
        '--output-dir',
        default=os.environ.get('LLM_HUMAN_RELAY_IMG_DIR') or None,
        help='directory where camera snapshots are saved',
    )

    args = parser.parse_args()

    node = OperationSide(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
    )

    node.loop()


if __name__ == '__main__':
    main()
