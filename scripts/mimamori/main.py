import argparse
import json
import subprocess
import threading
import time
from subprocess import TimeoutExpired
from typing import Final

import roslibpy

try:
    from .seek_openduck import OperationSide
    from ..ros2_base import Ros2Base
except ImportError:
    from seek_openduck import OperationSide
    from ros2_base import Ros2Base

# ここでインタプリタとかエントリポイントの指定してるので、これを変えないといけないです。
# https://github.com/fabc-why/whisper
WHISPER_PYTHON: Final = '/home/openduck/openduck_function/whisper/.venv/bin/python'
WHISPER_MAIN: Final = '/home/openduck/openduck_function/whisper/src/main.py'
WHISPER_CWD: Final = '/home/openduck/openduck_function/whisper'
TOPIC_NAME: Final = 'openduck/whisper_heard'
MESSAGE_TYPE: Final = 'std_msgs/String'
ROBOT_ID: Final = '1'
STOP_COMMAND: Final = [0.0] * 7


class Mimamori:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 9090,
        whisper_python: str = WHISPER_PYTHON,
        whisper_main: str = WHISPER_MAIN,
    ):
        self.host = host
        self.port = port
        self.whisper_process = subprocess.Popen(
            [whisper_python, whisper_main],
            cwd=WHISPER_CWD,
        )

        try:
            self.client = roslibpy.Ros(host=host, port=port)
            self.client.run()

            self.subscription = roslibpy.Topic(
                self.client,
                TOPIC_NAME,
                MESSAGE_TYPE,
            )
            self.subscription.subscribe(self.heard_callback)
            self.command_base = Ros2Base(
                ros_namespace=f'openduck{ROBOT_ID}',
                client=self.client,
                image_callback=lambda frame: None,
            )
        except Exception:
            if self.whisper_process.poll() is None:
                self.whisper_process.terminate()
                self.whisper_process.wait()
            raise

        self.running = True
        self.seek_node: OperationSide | None = None
        self.seek_thread: threading.Thread | None = None
        self.seek_lock = threading.Lock()
        self.seek_starting = False

        print(f'Subscribed to {TOPIC_NAME}')
        print(f'Started whisper process: {whisper_main}')

    def heard_callback(self, message):
        raw_data = message.get('data', '')
        print(f'Whisper heard: {raw_data}')

        try:
            payload = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        command = str(payload.get('meaning', '')).strip().lower()

        if command == 'seek':
            self.start_seek()
        elif command == 'stop':
            self.stop_seek()

    def start_seek(self):
        with self.seek_lock:
            if self.seek_starting:
                return
            if self.seek_thread is not None and self.seek_thread.is_alive():
                return
            self.seek_starting = True

        threading.Thread(
            target=self._start_seek,
            name='start-seek-openduck',
            daemon=True,
        ).start()

    def _start_seek(self):
        try:
            seek_node = OperationSide(
                host=self.host,
                port=self.port,
                start_auto=True,
                ros_namespace=f'openduck{ROBOT_ID}',
                enable_keyboard=False,
                ros_client=self.client,
            )
            seek_thread = threading.Thread(
                target=seek_node.loop,
                name='seek-openduck',
                daemon=True,
            )

            with self.seek_lock:
                if not self.running:
                    seek_node.cleanup(send_stop=False)
                    return
                self.seek_node = seek_node
                self.seek_thread = seek_thread
                seek_thread.start()
                print('Seek mode started')
        except Exception as error:
            print(f'Failed to start seek mode: {error}')
        finally:
            with self.seek_lock:
                self.seek_starting = False

    def stop_seek(self):
        with self.seek_lock:
            self.command_base.publish_command(STOP_COMMAND)

            if self.seek_starting:
                print('Seek mode is still starting')
                return
            if self.seek_node is None:
                return

            self.seek_node.cleanup(send_stop=False)
            self.seek_node = None
            self.seek_thread = None
            print('Seek mode stopped')

    def loop(self):
        try:
            while self.running and self.client.is_connected:
                if self.whisper_process.poll() is not None:
                    print(
                        'Whisper process exited with '
                        f'code {self.whisper_process.returncode}'
                    )
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            self.cleanup()

    def cleanup(self):
        if not self.running:
            return

        self.running = False
        self.stop_seek()
        self.subscription.unsubscribe()
        self.command_base.cleanup(send_stop=False)
        self.client.terminate()

        if self.whisper_process.poll() is None:
            self.whisper_process.terminate()
            try:
                self.whisper_process.wait(timeout=5)
            except TimeoutExpired:
                self.whisper_process.kill()
                self.whisper_process.wait()

        print('Mimamori shutdown complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--whisper-python', default=WHISPER_PYTHON)
    parser.add_argument('--whisper-main', default=WHISPER_MAIN)
    args = parser.parse_args()

    Mimamori(
        host=args.host,
        port=args.port,
        whisper_python=args.whisper_python,
        whisper_main=args.whisper_main,
    ).loop()


if __name__ == '__main__':
    main()
