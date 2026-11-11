import base64
import binascii
import json
import threading

import cv2
import numpy as np
import roslibpy


class Ros2Base:
    """共通のROS画像購読・コマンド送信機能を提供する基底クラス。"""

    def __init__(
        self,
        host='localhost',
        port=9090,
        ros_namespace='openduck',
        image_callback=None,
        stop_command=None,
        client=None,
    ):
        ros_namespace = ros_namespace.strip('/')
        if not ros_namespace:
            raise ValueError('ros_namespace must not be empty')

        self.client = client or roslibpy.Ros(host=host, port=port)
        self._owns_client = client is None
        if self._owns_client:
            self.client.run()
        self.running = True
        self.lock = threading.Lock()
        self.stop_command = stop_command
        self.image_callback = image_callback or self.handle_image
        self._cleaned_up = False

        self.sub = roslibpy.Topic(
            self.client,
            f'{ros_namespace}/head_cam/compressed',
            'sensor_msgs/CompressedImage'
        )
        self.sub.subscribe(self._image_callback)

        self.pub = roslibpy.Topic(
            self.client,
            f'{ros_namespace}/commands',
            'std_msgs/String'
        )
        self.pub.advertise()

    def handle_image(self, frame):
        """デコード済み画像を処理するための拡張ポイント。"""

    def _image_callback(self, message):
        try:
            frame = self.decode_compressed_image(message)
            self.image_callback(frame)
        except (ValueError, TypeError, binascii.Error, cv2.error) as error:
            print(f'Error processing image message: {error}')

    def decode_compressed_image(self, message):
        data = message.get('data')
        if data is None:
            raise ValueError('Image message has no data')

        if isinstance(data, str):
            image_bytes = base64.b64decode(data)
            array = np.frombuffer(image_bytes, dtype=np.uint8)
        elif isinstance(data, list):
            array = np.array(data, dtype=np.uint8)
        else:
            raise TypeError(f'Unsupported image data type: {type(data)}')

        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Failed to decode image')
        return frame

    def publish_command(self, command):
        message = roslibpy.Message({'data': json.dumps(command)})
        self.pub.publish(message)

    def cleanup(self, send_stop=True):
        if self._cleaned_up:
            return

        self._cleaned_up = True
        self.running = False

        if send_stop and self.stop_command is not None:
            self.publish_command(self.stop_command)

        self.sub.unsubscribe()
        self.pub.unadvertise()
        if self._owns_client:
            self.client.terminate()