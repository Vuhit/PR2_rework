"""
 * Copyright 1996-2021 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description: sample controller for the PR2
 */
"""
import sys
import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import math

from controller import Camera
from controller import Device
from controller import InertialUnit
from controller import Motor
from controller import PositionSensor
from controller import Robot
from controller import TouchSensor

from enum import Enum, auto


TIME_STEP = 16

#PR2 constants
MAX_WHEEL_SPEED = 3.0        # maximum velocity for the wheels [rad / s]
WHEELS_DISTANCE = 0.4492     # distance between 2 caster wheels (the four wheels are located in square) [m]
SUB_WHEELS_DISTANCE = 0.098  # distance between 2 sub wheels of a caster wheel [m]
WHEEL_RADIUS = 0.08          # wheel radius

# function to check if a double is almost equal to another
TOLERANCE = 0.05  # arbitrary value
def ALMOST_EQUAL(a, b):
    return ((a < b + TOLERANCE) and (a > b - TOLERANCE))

# helper constants to distinguish the motors
class Wheels(Enum):
    FLL_WHEEL = 0
    FLR_WHEEL = auto()
    FRL_WHEEL = auto()
    FRR_WHEEL = auto()
    BLL_WHEEL = auto()
    BLR_WHEEL = auto()
    BRL_WHEEL = auto()
    BRR_WHEEL = auto()

class Rotation(Enum):
    FL_ROTATION = 0
    FR_ROTATION = auto()
    BL_ROTATION = auto()
    BR_ROTATION = auto()

class Shoulder(Enum):
    SHOULDER_ROLL = 0
    SHOULDER_LIFT = auto()
    UPPER_ARM_ROLL = auto()
    ELBOW_LIFT = auto()
    WRIST_ROLL = auto()

class Finger(Enum):
    LEFT_FINGER = 0
    RIGHT_FINGER = auto()
    LEFT_TIP = auto()
    RIGHT_TIP = auto()

# PR2 motors and their sensors
class Devices:

    wheel_motors = [None] * 8
    wheel_sensors = [None] * 8
    rotation_motors = [None] * 4
    rotation_sensors = [None] * 4
    left_arm_motors = [None] * 5
    left_arm_sensors = [None] * 5
    right_arm_motors = [None] * 5
    right_arm_sensors = [None] * 5
    right_finger_motors = [None] * 4
    right_finger_sensors = [None] * 4
    left_finger_motors = [None] * 4
    left_finger_sensors = [None] * 4
    head_tilt_motor = None
    torso_motor = None
    torso_sensor = None
    
    # Sensors
    left_finger_contact_sensors = [None] * 2
    right_finger_contact_sensors = [None] * 2
    imu_sensor = None
    wide_stereo_l_stereo_camera_sensor = None
    wide_stereo_r_stereo_camera_sensor = None
    high_def_sensor = None
    r_forearm_cam_sensor = None
    l_forearm_cam_sensor = None
    laser_tilt = None
    base_laser = None
    



# Simpler step function
def step():
    if robot.step(TIME_STEP) == -1:
        exit()
        

# Retrieve all the pointers to the PR2 devices
def initialize_devices():
    Devices.wheel_motors[Wheels.FLL_WHEEL.value] = robot.getDevice("fl_caster_l_wheel_joint")
    Devices.wheel_motors[Wheels.FLR_WHEEL.value] = robot.getDevice("fl_caster_r_wheel_joint")
    Devices.wheel_motors[Wheels.FRL_WHEEL.value] = robot.getDevice("fr_caster_l_wheel_joint")
    Devices.wheel_motors[Wheels.FRR_WHEEL.value] = robot.getDevice("fr_caster_r_wheel_joint")
    Devices.wheel_motors[Wheels.BLL_WHEEL.value] = robot.getDevice("bl_caster_l_wheel_joint")
    Devices.wheel_motors[Wheels.BLR_WHEEL.value] = robot.getDevice("bl_caster_r_wheel_joint")
    Devices.wheel_motors[Wheels.BRL_WHEEL.value] = robot.getDevice("br_caster_l_wheel_joint")
    Devices.wheel_motors[Wheels.BRR_WHEEL.value] = robot.getDevice("br_caster_r_wheel_joint")

    
    for i in range(Wheels.BRR_WHEEL.value+1):
        Devices.wheel_sensors[i] = Devices.wheel_motors[i].getPositionSensor()

    Devices.rotation_motors[Rotation.FL_ROTATION.value] = robot.getDevice("fl_caster_rotation_joint")
    Devices.rotation_motors[Rotation.FR_ROTATION.value] = robot.getDevice("fr_caster_rotation_joint")
    Devices.rotation_motors[Rotation.BL_ROTATION.value] = robot.getDevice("bl_caster_rotation_joint")
    Devices.rotation_motors[Rotation.BR_ROTATION.value] = robot.getDevice("br_caster_rotation_joint")
    
    
    for i in range(Rotation.BR_ROTATION.value+1):
        Devices.rotation_sensors[i] = Devices.rotation_motors[i].getPositionSensor()

    Devices.left_arm_motors[Shoulder.SHOULDER_ROLL.value] = robot.getDevice("l_shoulder_pan_joint")
    Devices.left_arm_motors[Shoulder.SHOULDER_LIFT.value] = robot.getDevice("l_shoulder_lift_joint")
    Devices.left_arm_motors[Shoulder.UPPER_ARM_ROLL.value] = robot.getDevice("l_upper_arm_roll_joint")
    Devices.left_arm_motors[Shoulder.ELBOW_LIFT.value] = robot.getDevice("l_elbow_flex_joint")
    Devices.left_arm_motors[Shoulder.WRIST_ROLL.value] = robot.getDevice("l_wrist_roll_joint")
    
    
    for i in range(Shoulder.WRIST_ROLL.value+1):
        Devices.left_arm_sensors[i] = Devices.left_arm_motors[i].getPositionSensor()

    Devices.right_arm_motors[Shoulder.SHOULDER_ROLL.value] = robot.getDevice("r_shoulder_pan_joint");
    Devices.right_arm_motors[Shoulder.SHOULDER_LIFT.value] = robot.getDevice("r_shoulder_lift_joint");
    Devices.right_arm_motors[Shoulder.UPPER_ARM_ROLL.value] = robot.getDevice("r_upper_arm_roll_joint");
    Devices.right_arm_motors[Shoulder.ELBOW_LIFT.value] = robot.getDevice("r_elbow_flex_joint");
    Devices.right_arm_motors[Shoulder.WRIST_ROLL.value] = robot.getDevice("r_wrist_roll_joint");
  
    for i in range(Shoulder.WRIST_ROLL.value+1):
        Devices.right_arm_sensors[i] = Devices.right_arm_motors[i].getPositionSensor()

    Devices.left_finger_motors[Finger.LEFT_FINGER.value] = robot.getDevice("l_gripper_l_finger_joint")
    Devices.left_finger_motors[Finger.RIGHT_FINGER.value] = robot.getDevice("l_gripper_r_finger_joint")
    Devices.left_finger_motors[Finger.LEFT_TIP.value] = robot.getDevice("l_gripper_l_finger_tip_joint")
    Devices.left_finger_motors[Finger.RIGHT_TIP.value] = robot.getDevice("l_gripper_r_finger_tip_joint")
    
    for i in range(Finger.RIGHT_TIP.value+1):
        Devices.left_finger_sensors[i] = Devices.left_finger_motors[i].getPositionSensor()

    Devices.right_finger_motors[Finger.LEFT_FINGER.value] = robot.getDevice("r_gripper_l_finger_joint")
    Devices.right_finger_motors[Finger.RIGHT_FINGER.value] = robot.getDevice("r_gripper_r_finger_joint")
    Devices.right_finger_motors[Finger.LEFT_TIP.value] = robot.getDevice("r_gripper_l_finger_tip_joint")
    Devices.right_finger_motors[Finger.RIGHT_TIP.value] = robot.getDevice("r_gripper_r_finger_tip_joint")
    
    
    for i in range(Finger.RIGHT_TIP.value+1):
        Devices.right_finger_sensors[i] = Devices.right_finger_motors[i].getPositionSensor()

    Devices.head_tilt_motor = robot.getDevice("head_tilt_joint")
    Devices.torso_motor = robot.getDevice("torso_lift_joint")
    Devices.torso_sensor = robot.getDevice("torso_lift_joint_sensor")

    Devices.left_finger_contact_sensors[Finger.LEFT_FINGER.value] = robot.getDevice("l_gripper_l_finger_tip_contact_sensor")
    Devices.left_finger_contact_sensors[Finger.RIGHT_FINGER.value] = robot.getDevice("l_gripper_r_finger_tip_contact_sensor")
    Devices.right_finger_contact_sensors[Finger.LEFT_FINGER.value] = robot.getDevice("r_gripper_l_finger_tip_contact_sensor")
    Devices.right_finger_contact_sensors[Finger.RIGHT_FINGER.value] = robot.getDevice("r_gripper_r_finger_tip_contact_sensor")

    Devices.imu_sensor = robot.getDevice("imu_sensor")

    Devices.wide_stereo_l_stereo_camera_sensor = robot.getDevice("wide_stereo_l_stereo_camera_sensor")
    Devices.wide_stereo_r_stereo_camera_sensor = robot.getDevice("wide_stereo_r_stereo_camera_sensor")
    Devices.high_def_sensor = robot.getDevice("high_def_sensor")
    Devices.r_forearm_cam_sensor = robot.getDevice("r_forearm_cam_sensor")
    Devices.l_forearm_cam_sensor = robot.getDevice("l_forearm_cam_sensor")
    Devices.laser_tilt = robot.getDevice("laser_tilt")
    Devices.base_laser = robot.getDevice("base_laser")
    
    return

# enable the robot devices
def enable_devices():
    for i in range(8):
        Devices.wheel_sensors[i].enable(TIME_STEP)
   
    # init the motors for speed control
        Devices.wheel_motors[i].setPosition(float("inf"))
        Devices.wheel_motors[i].setVelocity(0.0)
        

    for i in range(4):
        Devices.rotation_sensors[i].enable(TIME_STEP)

    for i in range(2):
        Devices.left_finger_contact_sensors[i].enable(TIME_STEP)
        Devices.right_finger_contact_sensors[i].enable(TIME_STEP)

    for i in range(4):
        Devices.left_finger_sensors[i].enable(TIME_STEP)
        Devices.right_finger_sensors[i].enable(TIME_STEP)

    for i in range(5):
        Devices.left_arm_sensors[i].enable(TIME_STEP)
        Devices.right_arm_sensors[i].enable(TIME_STEP)

    Devices.torso_sensor.enable(TIME_STEP)
    
    return
    
"""
  the following devices are not used in this simulation.

  wb_inertial_unit_enable(imu_sensor, TIME_STEP);
  wb_camera_enable(wide_stereo_l_stereo_camera_sensor, TIME_STEP);
  wb_camera_enable(wide_stereo_r_stereo_camera_sensor, TIME_STEP);
  wb_camera_enable(high_def_sensor, TIME_STEP);
  wb_camera_enable(r_forearm_cam_sensor, TIME_STEP);
  wb_camera_enable(l_forearm_cam_sensor, TIME_STEP);
  wb_lidar_enable(laser_tilt, TIME_STEP);
  wb_lidar_enable(base_laser, TIME_STEP);
  
"""


# set the speeds of the robot wheels
def set_wheels_speeds(fll, flr, frl, frr, bll, blr, brl, brr):
    
    #print(Devices.wheel_motors[Wheels.FLL_WHEEL.value].getVelocity())

    Devices.wheel_motors[Wheels.FLL_WHEEL.value].setVelocity(fll)
    Devices.wheel_motors[Wheels.FLR_WHEEL.value].setVelocity(flr)
    Devices.wheel_motors[Wheels.FRL_WHEEL.value].setVelocity(frl)
    Devices.wheel_motors[Wheels.FRR_WHEEL.value].setVelocity(frr)
    Devices.wheel_motors[Wheels.BLL_WHEEL.value].setVelocity(bll)
    Devices.wheel_motors[Wheels.BLR_WHEEL.value].setVelocity(blr)
    Devices.wheel_motors[Wheels.BRL_WHEEL.value].setVelocity(brl)
    Devices.wheel_motors[Wheels.BRR_WHEEL.value].setVelocity(brr)
    
    #print(Devices.wheel_motors[Wheels.FLL_WHEEL.value].getVelocity())
    

    return
    
    
def set_wheels_speed(speed):
    set_wheels_speeds(speed, speed, speed, speed, speed, speed, speed, speed)
    return

def stop_wheels():
    set_wheels_speeds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return

# enable/disable the torques on the wheels motors
def enable_passive_wheels(enable):
    torques = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if (enable):
        for i in range(8):
            torques[i] = Devices.wheel_motors[i].getAvailableTorque
            Devices.wheel_motors[i].setAvailableTorque(0.0)
    else:
        for i in range(8):
            Devices.wheel_motors[i].setAvailableTorque(torques[i])
    
    return



# Set the rotation wheels angles.
# If wait_on_feedback is true, the function is left when the rotational motors have reached their target positions.
def set_rotation_wheels_angles(fl, fr, bl, br, wait_on_feedback):
    
    if (wait_on_feedback):
        stop_wheels()
        enable_passive_wheels(True)

    Devices.rotation_motors[Rotation.FL_ROTATION.value].setPosition(fl)
    Devices.rotation_motors[Rotation.FR_ROTATION.value].setPosition(fr)
    Devices.rotation_motors[Rotation.BL_ROTATION.value].setPosition(bl)
    Devices.rotation_motors[Rotation.BR_ROTATION.value].setPosition(br)
    
    
    #for i in range(4):
        #print(Devices.rotation_sensors[i].getValue())
        
    if (wait_on_feedback):
        target = [fl, fr, bl, br]

        while (True):
            all_reached = True
            for i in range(4):
                #print("sensor "+ str(i) + " " + str(Devices.rotation_sensors[i].getValue()))
                current_position = Devices.rotation_sensors[i].getValue()
                if (not ALMOST_EQUAL(current_position, target[i])):
                    all_reached = False
                    break

            if (all_reached):
                break
            else:
                step()

        enable_passive_wheels(False)

    return

# High level function to rotate the robot around itself of a given angle [rad]
# Note: the angle can be negative
def robot_rotate(angle):
    stop_wheels()
    
    set_rotation_wheels_angles(3.0 * (math.pi/4), (math.pi/4), -3.0 * (math.pi/4), (-math.pi/4), True)
    
    
    #set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, True)
    
    
    max_wheel_speed = 0.0
    if angle > 0:
        max_wheel_speed = MAX_WHEEL_SPEED
    else:
        max_wheel_speed = -MAX_WHEEL_SPEED
    
    #print(max_wheel_speed)
    
    set_wheels_speed(max_wheel_speed)
    
    
    #while(True):
        #step()
    
    initial_wheel0_position = Devices.wheel_sensors[Wheels.FLL_WHEEL.value].getValue()
    # expected travel distance done by the wheel
    expected_travel_distance = math.fabs(angle * 0.5 * (WHEELS_DISTANCE + SUB_WHEELS_DISTANCE))

    while (True):
        wheel0_position = Devices.wheel_sensors[Wheels.FLL_WHEEL.value].getValue()
    # travel distance done by the wheel
        wheel0_travel_distance = math.fabs(WHEEL_RADIUS * (wheel0_position - initial_wheel0_position))

        if (wheel0_travel_distance > expected_travel_distance):
            break

    # reduce the speed before reaching the target
        if (expected_travel_distance - wheel0_travel_distance < 0.025):
            set_wheels_speed(0.1 * max_wheel_speed)
        
        step()
        
  

    # reset wheels
    set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, True)
    stop_wheels()
    return


# High level function to go forward for a given distance [m]
# Note: the distance can be negative
def robot_go_forward(distance):

    max_wheel_speed = 0.0
    if distance > 0:
        max_wheel_speed = MAX_WHEEL_SPEED
    else: 
        max_wheel_speed = -MAX_WHEEL_SPEED
        
    set_wheels_speed(max_wheel_speed)

    initial_wheel0_position = Devices.wheel_sensors[Wheels.FLL_WHEEL.value].getValue()

    while (True):
        wheel0_position = Devices.wheel_sensors[Wheels.FLL_WHEEL.value].getValue()
        # travel distance done by the wheel
        wheel0_travel_distance = math.fabs(WHEEL_RADIUS * (wheel0_position - initial_wheel0_position))

        if (wheel0_travel_distance > math.fabs(distance)):
            break;

        # reduce the speed before reaching the target
        if ((math.fabs(distance) - wheel0_travel_distance) < 0.025):
            set_wheels_speed(0.1 * max_wheel_speed)

        step()


    stop_wheels()
    return


# Open or close the gripper.
# If wait_on_feedback is true, the gripper is stopped either when the target is reached,
# or either when something has been gripped
def set_gripper(left, open, torqueWhenGripping, wait_on_feedback):

    motors = [None] * 4
    motors[Finger.LEFT_FINGER.value] = None
    if left:
        motors[Finger.LEFT_FINGER.value] = Devices.left_finger_motors[Finger.LEFT_FINGER.value]
    else:
        motors[Finger.LEFT_FINGER.value] = Devices.right_finger_motors[Finger.LEFT_FINGER.value]
  
    motors[Finger.RIGHT_FINGER.value] = None
  
    if left:
        motors[Finger.RIGHT_FINGER.value] = Devices.left_finger_motors[Finger.RIGHT_FINGER.value]
    else:
        motors[Finger.RIGHT_FINGER.value] = Devices.right_finger_motors[Finger.RIGHT_FINGER.value]
      
    motors[Finger.LEFT_TIP.value] = None
    if left:
       motors[Finger.LEFT_TIP.value] = Devices.left_finger_motors[Finger.LEFT_TIP.value]
    else:
        motors[Finger.LEFT_TIP.value] = Devices.right_finger_motors[Finger.LEFT_TIP.value]
      
    motors[Finger.RIGHT_TIP.value] = None
    if left:
        motors[Finger.RIGHT_TIP.value] = Devices.left_finger_motors[Finger.RIGHT_TIP.value]
    else:
        motors[Finger.RIGHT_TIP.value] = Devices.right_finger_motors[Finger.RIGHT_TIP.value]

    sensors = [None] * 4
    sensors[Finger.LEFT_FINGER.value] = None
    if left:
        sensors[Finger.LEFT_FINGER.value] = Devices.left_finger_sensors[Finger.LEFT_FINGER.value]
    else:
        sensors[Finger.LEFT_FINGER.value] = Devices.right_finger_sensors[Finger.LEFT_FINGER.value]
  
    sensors[Finger.RIGHT_FINGER.value] = None
    if left:
        sensors[Finger.RIGHT_FINGER.value] = Devices.left_finger_sensors[Finger.RIGHT_FINGER.value]
    else:
        sensors[Finger.RIGHT_FINGER.value] = Devices.right_finger_sensors[Finger.RIGHT_FINGER.value]
  
    sensors[Finger.LEFT_TIP.value] = None
    if left:
        sensors[Finger.LEFT_TIP.value] = Devices.left_finger_sensors[Finger.LEFT_TIP.value]
    else:
        sensors[Finger.LEFT_TIP.value] = Devices.right_finger_sensors[Finger.LEFT_TIP.value]
  
    sensors[Finger.RIGHT_TIP.value] = None
    if left:
        sensors[Finger.RIGHT_TIP.value] = Devices.left_finger_sensors[Finger.RIGHT_TIP.value]
    else:
        sensors[Finger.RIGHT_TIP.value] = Devices.right_finger_sensors[Finger.RIGHT_TIP.value]

    contacts = [None] * 2
    contacts[Finger.LEFT_FINGER.value] = None
    if left:
        contacts[Finger.LEFT_FINGER.value] = Devices.left_finger_contact_sensors[Finger.LEFT_FINGER.value]
    else:
        contacts[Finger.LEFT_FINGER.value] = Devices.right_finger_contact_sensors[Finger.LEFT_FINGER.value]
  
    contacts[Finger.RIGHT_FINGER.value] = None
    if left:
        contacts[Finger.RIGHT_FINGER.value] = Devices.left_finger_contact_sensors[Finger.RIGHT_FINGER.value]
    else:
        contacts[Finger.RIGHT_FINGER.value] = Devices.right_finger_contact_sensors[Finger.RIGHT_FINGER.value]

    firstCall = True
    maxTorque = 0.0
  
    if (firstCall):
        maxTorque = motors[Finger.LEFT_FINGER.value].getAvailableTorque()
        firstCall = False


    for i in range (4):
        motors[i].setAvailableTorque(maxTorque)

    if (open):
        targetOpenValue = 0.5
        for i in range(4):
            motors[i].setPosition(targetOpenValue)

        if (wait_on_feedback):
            while (not ALMOST_EQUAL(sensors[Finger.LEFT_FINGER.value].getValue(), targetOpenValue)):
                step()
    
    else:
        targetCloseValue = 0.0
        for i in range(4):
            motors[i].setPosition(targetCloseValue)

        if (wait_on_feedback):
            # wait until the 2 touch sensors are fired or the target value is reached
            while (
        (contacts[Finger.LEFT_FINGER.value].getValue() == 0.0) or contacts[Finger.RIGHT_FINGER.value].getValue() == 0.0 and
        not ALMOST_EQUAL(sensors[Finger.LEFT_FINGER.value].getValue(), targetCloseValue)):
                step()
      
            current_position = sensors[Finger.LEFT_FINGER.value].getValue()
            for i in range(4):
                motors[i].setAvailableTorque(torqueWhenGripping)
                motors[i].setPosition(max(0.0, 0.95 * current_position))

    return

# Set the right arm position (forward kinematics)
# If wait_on_feedback is enabled, the function is left when the target is reached.
def set_right_arm_position(shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift,
                                   wrist_roll, wait_on_feedback):
                                   
    Devices.right_arm_motors[Shoulder.SHOULDER_ROLL.value].setPosition(shoulder_roll)
    Devices.right_arm_motors[Shoulder.SHOULDER_LIFT.value].setPosition(shoulder_lift)
    Devices.right_arm_motors[Shoulder.UPPER_ARM_ROLL.value].setPosition(upper_arm_roll)
    Devices.right_arm_motors[Shoulder.ELBOW_LIFT.value].setPosition(elbow_lift)
    Devices.right_arm_motors[Shoulder.WRIST_ROLL.value].setPosition(wrist_roll)

    if (wait_on_feedback):
        while (not ALMOST_EQUAL(Devices.right_arm_sensors[Shoulder.SHOULDER_ROLL.value].getValue(), shoulder_roll) or
           not ALMOST_EQUAL(Devices.right_arm_sensors[Shoulder.SHOULDER_LIFT.value].getValue(), shoulder_lift) or
           not ALMOST_EQUAL(Devices.right_arm_sensors[Shoulder.UPPER_ARM_ROLL.value].getValue(), upper_arm_roll) or
           not ALMOST_EQUAL(Devices.right_arm_sensors[Shoulder.ELBOW_LIFT.value].getValue(), elbow_lift) or
           not ALMOST_EQUAL(Devices.right_arm_sensors[Shoulder.WRIST_ROLL.value].getValue(), wrist_roll)):
            step()
            


# Idem for the left arm
def set_left_arm_position(shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift,
                                  wrist_roll, wait_on_feedback):
    Devices.left_arm_motors[Shoulder.SHOULDER_ROLL.value].setPosition(shoulder_roll)
    Devices.left_arm_motors[Shoulder.SHOULDER_LIFT.value].setPosition(shoulder_lift)
    Devices.left_arm_motors[Shoulder.UPPER_ARM_ROLL.value].setPosition(upper_arm_roll)
    Devices.left_arm_motors[Shoulder.ELBOW_LIFT.value].setPosition(elbow_lift)
    Devices.left_arm_motors[Shoulder.WRIST_ROLL.value].setPosition(wrist_roll)

    if (wait_on_feedback):
        while (not ALMOST_EQUAL(Devices.left_arm_sensors[Shoulder.SHOULDER_ROLL.value].getValue(), shoulder_roll) or
           not ALMOST_EQUAL(Devices.left_arm_sensors[Shoulder.SHOULDER_LIFT.value].getValue(), shoulder_lift) or
           not ALMOST_EQUAL(Devices.left_arm_sensors[Shoulder.UPPER_ARM_ROLL.value].getValue(), upper_arm_roll) or
           not ALMOST_EQUAL(Devices.left_arm_sensors[Shoulder.ELBOW_LIFT.value].getValue(), elbow_lift) or
           not ALMOST_EQUAL(Devices.left_arm_sensors[Shoulder.WRIST_ROLL.value].getValue(), wrist_roll)):
            step()



# Set the torso height
# If wait_on_feedback is enabled, the function is left when the target is reached.
def set_torso_height(height, wait_on_feedback):
    Devices.torso_motor.setPosition(height)

    if (wait_on_feedback):
        while (not ALMOST_EQUAL(Devices.torso_sensor.getValue(), height)):
            step()

    return
# Convenient initial position
def set_initial_position():
    set_left_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, False)
    set_right_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, False)

    set_gripper(False, True, 0.0, False)
    set_gripper(True, True, 0.0, False)

    set_torso_height(0.2, True)
    
    return


robot = Robot()

initialize_devices()
enable_devices()
set_initial_position()

#go to the initial position
set_left_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
set_right_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
robot_go_forward(0.35)


# main loop
while (True):
    # close the gripper with forcefeedback
    set_gripper(True, False, 20.0, True)
    set_gripper(False, False, 20.0, True)
    # lift the arms
    set_left_arm_position(0.0, 0.5, 0.0, -1.0, 0.0, True)
    set_right_arm_position(0.0, 0.5, 0.0, -1.0, 0.0, True)
    # go to the other table
    robot_go_forward(-0.35)
    robot_rotate(math.pi)
    robot_go_forward(0.35)
    # move the arms down
    set_left_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
    set_right_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
    # open the grippers
    set_gripper(True, True, 0.0, True)
    set_gripper(False, True, 0.0, True)


