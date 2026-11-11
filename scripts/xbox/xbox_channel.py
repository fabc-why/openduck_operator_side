import cv2
import argparse
import time
import pygame

try:
    from scripts.ros2_base import Ros2Base
except ModuleNotFoundError as error:
    if error.name != 'scripts':
        raise
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ros2_base import Ros2Base


def positive_int(value):
    robot_id = int(value)
    if robot_id < 0:
        raise argparse.ArgumentTypeError('robot_id must be a non-negative integer')
    return robot_id


class OperationSide(Ros2Base):
    def __init__(self, host='localhost', port=9090, ros_namespace='openduck'):
        self.command = [0.0] * 7
        super().__init__(
            host=host,
            port=port,
            ros_namespace=ros_namespace,
            image_callback=self.handle_image,
            stop_command=self.command,
        )

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError('Xbox controller が見つかりません')

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print('OperationSide started')
        print(f'Controller: {self.joystick.get_name()}')
        print(f'axes: {self.joystick.get_numaxes()}')
        print('axis 1: forward/backward')
        print('axis 0: left/right')
        print('axis 3: yaw')
        print('Ctrl+C to quit')

    def deadzone(self, value, threshold=0.15):
        if abs(value) < threshold:
            return 0.0
        return value

    def publish_controller_command(self):
        pygame.event.pump()

        axis_0_raw = self.joystick.get_axis(0)
        axis_1_raw = self.joystick.get_axis(1)
        axis_2_raw = self.joystick.get_axis(2)
        axis_3_raw = self.joystick.get_axis(3)
        axis_4_raw = self.joystick.get_axis(4)
        axis_5_raw = self.joystick.get_axis(5)

        button_x = self.joystick.get_button(2)  # Xボタン
        
        dpad = self.joystick.get_hat(0)  # D-pad

        button_lb = self.joystick.get_button(4)  # LBボタン
        button_rb = self.joystick.get_button(5)  # RBボタン


        lstick_x = self.deadzone(axis_0_raw) #左のスティック左右
        lstick_y = self.deadzone(axis_1_raw) #左のスティック上下
        ltrigger = self.deadzone(axis_2_raw) #左のトリガー
        rstick_x = self.deadzone(axis_3_raw) #右のスティック左右
        rstick_y = self.deadzone(axis_4_raw) #右のスティック上下
        rtrigger = self.deadzone(axis_5_raw) #右のトリガー

        max_forward = 0.3
        max_side = 0.5
        max_yaw = 1.0

        x_boost = 1.5 if button_x else 1.0  # Xボタンが押されている場合はスケールを1.0にする
        rb = -1 if button_rb else 1  # RBボタンが押されている場合はマイナス
        lb = -1 if button_lb else 1  # LBボタンが押されている場合はマイナス

        forward = -lstick_y * max_forward * x_boost #前進は左スティック上下
        side = -dpad[0] * max_side #左右は十字キー左右
        yaw = -lstick_x * max_yaw #旋回は左スティック左右

        neck1 = rstick_y * 5.0 #首上下は右スティック上下
        neck2 = rstick_x * 3.0 #首左右は右スティック左右
        neck3 = lb * (ltrigger + 1) / 2 * 10.0 #首前後は左トリガー
        neck4 = rb * (rtrigger + 1) / 2 * 5.0 #首前後は右トリガー

        self.command = [
            forward,
            side,
            yaw,
            neck1,
            neck2,
            neck3,
            neck4
        ]

        self.publish_command(self.command)

        print(
            f'Published command: {self.command} | '
            f'axes: {[rtrigger, ltrigger]} | '
            f'dpad: {dpad} | '
            f'button_x: {button_x}'
        )

    def handle_image(self, frame):
        try:
            cv2.imshow('operation_side', frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f'Error decoding/displaying image: {e}')

    def loop(self):
        rate = 20.0
        dt = 1.0 / rate

        try:
            while self.running and self.client.is_connected:
                self.publish_controller_command()
                time.sleep(dt)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        finally:
            self.cleanup()

    def cleanup(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

        super().cleanup()

        print('OperationSide shutdown complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'robot_id',
        nargs='?',
        type=positive_int,
        help='Numeric robot ID, for example 1 -> openduck1',
    )
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9090)
    args = parser.parse_args()

    node = OperationSide(
        host=args.host,
        port=args.port,
        ros_namespace=f'openduck{args.robot_id}'
        if args.robot_id is not None
        else 'openduck',
    )
    node.loop()


if __name__ == '__main__':
    main()
