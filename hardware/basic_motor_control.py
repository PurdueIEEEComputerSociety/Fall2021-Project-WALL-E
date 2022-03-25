import RPi.GPIO as GPIO
import time

# Test file

class MotorController():
    def __init__(self, left_motor_pin, right_motor_pin):
        self.left_motor_pin = left_motor_pin
        self.right_motor_pin = right_motor_pin

    def GPIO_setup(self):
        GPIO.setup(self.right_motor_pin, GPIO.OUT)
        GPIO.setup(self.left_motor_pin, GPIO.OUT)
        GPIO.output(self.left_motor_pin, GPIO.LOW)
        GPIO.output(self.right_motor_pin, GPIO.LOW)

    def _turn_left_time(self):
        tme = 0
        while tme <= 1:
            GPIO.PWM(self.right_motor_pin, GPIO.HIGH)
            tme += 0.001
            time.sleep(0.001)
        GPIO.output(self.right_motor_pin, GPIO.LOW)

    def _turn_right_time(self):
        tme = 0
        while tme <= 1:
            GPIO.PWM(self.left_motor_pin, GPIO.HIGH)
            tme += 0.001
            time.sleep(0.001)
        GPIO.output(self.left_motor_pin, GPIO.LOW)

    def _go_straight(self):
        tme = 0
        while tme <= 1:
            GPIO.PWM(self.left_motor_pin, GPIO.HIGH)
            GPIO.PWM(self.right_motor_pin, GPIO.HIGH)
            tme += 0.001
            time.sleep(0.001)
        GPIO.output(self.left_motor_pin, GPIO.LOW)
        GPIO.output(self.right_motor_pin, GPIO.LOW)

if __name__ == '__main__':
    channel1 = 1
    channel2 = 2
    motorController = MotorController(channel1, channel2)
    motorController.GPIO_setup()