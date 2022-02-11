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

    def stop_robot(self):
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

    def motor_control(self, forward_speed = 0.5, angular_speed = 0, mov_dir = 0):

        GPIO.PWM(self.left_motor_pin, mov_dir * forward_speed - angular_speed)
        GPIO.PWM(self.right_motor_pin, mov_dir * forward_speed + angular_speed)

    def get_dir(direction):
        # Use ROS/ anything else to get direction from computer vision algorithm
        if direction == 0:
            angular_speed = 0
        elif direction == 1:
            angular_speed = 0.2 # turn left
        else:
            angular_speed = - 0.2 # turn right
        return angular_speed

    def def_PID_vals(self):
        # There is only one plane that the robot can change its direction
        self.kp = 0
        self.kd = 0
        self.ki = 0
        self.dt = 0.004
        self.t = 0

    def PID_forward(self):
        # Need to add - calculate the forward speed based on the speed of a person
        # and the distance form a person
        pass

    def PID_angular(self, forward_speed):
        # Desired pos - angle of orientation (vector ?)
        # Actual pos - actual angle (vector ?) of orientation
        # curr_err - difference between two angles (vectors ?)
        # increasing angle difference between the desired and actual orientation increases error

        curr_error = desired_pos - actual_pos

        error_p = self.kp * (curr_err - prev_err)
        error_d = self.kd * (curr_err - prev_err) / self.dt
        error_i = self.ki * (curr_err - prev_err) * t

        error_total = error_p + error_i + error_d

        GPIO.PWM(self.left_motor_pin, forward_speed - error_total)
        GPIO.PWM(self.right_motor_pin, forward_speed + error_total)

        prev_err = curr_err
        t += self.dt
        t %= 1000000
        time.sleep(0.004)

if __name__ == '__main__':
    channel1 = 1
    channel2 = 2
    run = 0

    motorController = MotorController(channel1, channel2)

    motorController.GPIO_setup()
    motorController._go_straight(2, 0.8)

    while 1:
        if run == 1:
            angular_speed = motorController.getDir(0)
            motorController.motor_control(0.5, angular_speed, 1)
        else:
            motorController.stop_robot()
        time.sleep(0.004)
