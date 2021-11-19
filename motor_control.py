import RPi.GPIO as GPIO
import time

class MotorController():
    def __init__(self, left_motor_pin, right_motor_pin):
        self.left_motor_pin = left_motor_pin
        self.right_motor_pin = right_motor_pin

    def GPIO_setup(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.right_motor_pin, GPIO.OUT)
        GPIO.setup(self.left_motor_pin, GPIO.OUT)
        GPIO.output(self.left_motor_pin, GPIO.LOW)
        GPIO.output(self.right_motor_pin, GPIO.LOW)

    def _turn_left_time(self, desired_time, speed):
        tme = 0
        while tme <= desired_time:
            GPIO.PWM(self.right_motor_pin, speed)
            tme += 0.004
            time.sleep(0.004)
        GPIO.output(self.right_motor_pin, GPIO.LOW)

    def _turn_right_time(self, desired_time, speed):
        tme = 0
        while tme <= desired_time:
            GPIO.PWM(self.left_motor_pin, speed)
            tme += 0.004
            time.sleep(0.004)
        GPIO.output(self.left_motor_pin, GPIO.LOW)

    def _go_straight(self, desired_time, speed):
        tme = 0
        while tme <= desired_time:
            GPIO.PWM(self.left_motor_pin, speed)
            GPIO.PWM(self.right_motor_pin, speed)

            tme += 0.004
            print(f'going straight for: {tme}.')
            time.sleep(0.004)
        GPIO.output(self.left_motor_pin, GPIO.LOW)
        GPIO.output(self.right_motor_pin, GPIO.LOW)

    def motor_control(self, forward_speed = 0.5, angular_speed = 0, mov_dir):

        GPIO.PWM(self.left_motor_pin, mov_dir * forward_speed - angular_speed)
        GPIO.PWM(self.right_motor_pin, mov_dir * forward_speed + angular_speed)

    def getDir(direction):
        # Use ROS/ anything else to get direction from computer vision algorithm
        if direction = 0:
            angular_speed = 0
        elif direction = 1:
            angular_speed = 0.2 # turn left
        else:
            angular_speed = - 0.2 # turn right
        return angular_speed

if __name__ == '__main__':
    channel1 = 1
    channel2 = 2
    motorController = MotorController(channel1, channel2)
    motorController.GPIO_setup()

    motorController._go_straight(2, 0.8)

    while 1:
        angular_speed = motorController.getDir(0)
        motorController.motor_control(0.5, angular_speed, 1)
        time.sleep(0.004)
