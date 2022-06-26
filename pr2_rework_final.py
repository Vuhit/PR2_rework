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
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, SAC
#from stable_baselines import GAIL
#from stable_baselines.gail import ExpertDataset, generate_expert_traj
from keras.layers import (Conv2D, Flatten, Lambda, Dense, concatenate, Dropout, Input)
from keras.models import Model, load_model
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import math
import random
import pickle
from controller import Camera
from controller import Device
from controller import InertialUnit
from controller import Motor
from controller import PositionSensor
from controller import Robot
from controller import TouchSensor
from controller import Supervisor
from controller import Node






class ImitationGymEnviroment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()
        
        #testing on two motors, 1 right elbow flex , 2 is the right arm pan.
        #self.action_space = gym.spaces.Box(np.array([-2.29, -2.32]), np.array([0.71, 0.0]), dtype = np.float64)
        #self.action_space = gym.spaces.Box(np.array([-1.20, -2.10]), np.array([0.70, 0.0]), dtype = np.float64)
        self.action_space = gym.spaces.Box(np.array([-1.00, -1.00]), np.array([1.00, 1.00]), dtype = np.float32)

        self.observation_space = gym.spaces.Box(low = 0, high=255, shape = (140, 320, 3), dtype = np.uint8)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsTestEnv-v0', max_episode_steps=max_episode_steps)
        
        self.done = False
        self.TIME_STEP = 16
        self.STEP_COUNTER = 0
        
        self.normal_step_counter = 0
        
        self.correct_spot_count = 0
        #PR2 constants
        self.MAX_WHEEL_SPEED = 3.0        # maximum velocity for the wheels [rad / s]
        self.WHEELS_DISTANCE = 0.4492     # distance between 2 caster wheels (the four wheels are located in square) [m]
        self.SUB_WHEELS_DISTANCE = 0.098  # distance between 2 sub wheels of a caster wheel [m]
        self.WHEEL_RADIUS = 0.08          # wheel radius

        # function to check if a double is almost equal to another
        self.TOLERANCE = 0.05  # arbitrary value

        self.torques = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #wheel motors
        self.FLL_WHEEL = 0
        self.FLR_WHEEL = 1
        self.FRL_WHEEL = 2
        self.FRR_WHEEL = 3
        self.BLL_WHEEL = 4
        self.BLR_WHEEL = 5
        self.BRL_WHEEL = 6
        self.BRR_WHEEL = 7
    
        #Rotation motors
        self.FL_ROTATION = 0
        self.FR_ROTATION = 1
        self.BL_ROTATION = 2
        self.BR_ROTATION = 3
        
        #Shoulder motors
        self.SHOULDER_ROLL = 0
        self.SHOULDER_LIFT = 1
        self.UPPER_ARM_ROLL = 2
        self.ELBOW_LIFT = 3
        self.WRIST_ROLL = 4
        
        #Finger motors
        self.LEFT_FINGER = 0
        self.RIGHT_FINGER = 1
        self.LEFT_TIP = 2
        self.RIGHT_TIP = 3
        
        
        self.wheel_motors = [None] * 8
        self.wheel_sensors = [None] * 8
        self.rotation_motors = [None] * 4
        self.rotation_sensors = [None] * 4
        self.left_arm_motors = [None] * 5
        self.left_arm_sensors = [None] * 5
        self.right_arm_motors = [None] * 5
        self.right_arm_sensors = [None] * 5
        self.right_finger_motors = [None] * 4
        self.right_finger_sensors = [None] * 4
        self.left_finger_motors = [None] * 4
        self.left_finger_sensors = [None] * 4
        self.head_tilt_motor = None
        self.torso_motor = None
        self.torso_sensor = None
    
        self.head_camera = None
    
        # Sensors
        self.left_finger_contact_sensors = [None] * 2
        self.right_finger_contact_sensors = [None] * 2
        self.imu_sensor = None
        self.wide_stereo_l_stereo_camera_sensor = None
        self.wide_stereo_r_stereo_camera_sensor = None
        self.high_def_sensor = None
        self.r_forearm_cam_sensor = None
        self.l_forearm_cam_sensor = None
        self.laser_tilt = None
        self.base_laser = None
        self.external_camera = None
     

        
        #self.cookie_node = self.getFromDef("BBOX")
        
        self.can_node = self.getFromDef("CAN")
        
        self.marker_node = self.getFromDef("MARKER")
        self.marker_translation = self.marker_node.getField('translation')

        #self.image_list = None
        self.image_list = np.zeros((140,320,3))
        self.image_list = np.expand_dims(self.image_list, axis=0)
        
        self.third_person_image_list = np.zeros((140,320,3))
        self.third_person_image_list = np.expand_dims(self.third_person_image_list, axis=0)
        
        self.action_list = []
        self.x_action_list = []
        self.y_action_list = []
        
        self.total_reward_list = []
        
    def reset(self):
    
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.TIME_STEP)
        
        self.done= False
        self.STEP_COUNTER = 0
        self.correct_spot_count = 0
        self.normal_step_counter = 0


       
        
        self.wheel_motors[self.FLL_WHEEL] = self.getDevice("fl_caster_l_wheel_joint")
        self.wheel_motors[self.FLR_WHEEL] = self.getDevice("fl_caster_r_wheel_joint")
        self.wheel_motors[self.FRL_WHEEL] = self.getDevice("fr_caster_l_wheel_joint")
        self.wheel_motors[self.FRR_WHEEL] = self.getDevice("fr_caster_r_wheel_joint")
        self.wheel_motors[self.BLL_WHEEL] = self.getDevice("bl_caster_l_wheel_joint")
        self.wheel_motors[self.BLR_WHEEL] = self.getDevice("bl_caster_r_wheel_joint")
        self.wheel_motors[self.BRL_WHEEL] = self.getDevice("br_caster_l_wheel_joint")
        self.wheel_motors[self.BRR_WHEEL] = self.getDevice("br_caster_r_wheel_joint")

    
        for i in range(self.BRR_WHEEL+1):
            self.wheel_sensors[i] = self.wheel_motors[i].getPositionSensor()

        self.rotation_motors[self.FL_ROTATION] = self.getDevice("fl_caster_rotation_joint")
        self.rotation_motors[self.FR_ROTATION] = self.getDevice("fr_caster_rotation_joint")
        self.rotation_motors[self.BL_ROTATION] = self.getDevice("bl_caster_rotation_joint")
        self.rotation_motors[self.BR_ROTATION] = self.getDevice("br_caster_rotation_joint")
    
    
        for i in range(self.BR_ROTATION+1):
            self.rotation_sensors[i] = self.rotation_motors[i].getPositionSensor()

        self.left_arm_motors[self.SHOULDER_ROLL] = self.getDevice("l_shoulder_pan_joint")
        self.left_arm_motors[self.SHOULDER_LIFT] = self.getDevice("l_shoulder_lift_joint")
        self.left_arm_motors[self.UPPER_ARM_ROLL] = self.getDevice("l_upper_arm_roll_joint")
        self.left_arm_motors[self.ELBOW_LIFT] = self.getDevice("l_elbow_flex_joint")
        self.left_arm_motors[self.WRIST_ROLL] = self.getDevice("l_wrist_roll_joint")
    
    
        for i in range(self.WRIST_ROLL+1):
            self.left_arm_sensors[i] = self.left_arm_motors[i].getPositionSensor()

        self.right_arm_motors[self.SHOULDER_ROLL] = self.getDevice("r_shoulder_pan_joint");
        self.right_arm_motors[self.SHOULDER_LIFT] = self.getDevice("r_shoulder_lift_joint");
        self.right_arm_motors[self.UPPER_ARM_ROLL] = self.getDevice("r_upper_arm_roll_joint");
        self.right_arm_motors[self.ELBOW_LIFT] = self.getDevice("r_elbow_flex_joint");
        self.right_arm_motors[self.WRIST_ROLL] = self.getDevice("r_wrist_roll_joint");
  
        for i in range(self.WRIST_ROLL+1):
            self.right_arm_sensors[i] = self.right_arm_motors[i].getPositionSensor()

        self.left_finger_motors[self.LEFT_FINGER] = self.getDevice("l_gripper_l_finger_joint")
        self.left_finger_motors[self.RIGHT_FINGER] = self.getDevice("l_gripper_r_finger_joint")
        self.left_finger_motors[self.LEFT_TIP] = self.getDevice("l_gripper_l_finger_tip_joint")
        self.left_finger_motors[self.RIGHT_TIP] = self.getDevice("l_gripper_r_finger_tip_joint")
    
        for i in range(self.RIGHT_TIP+1):
            self.left_finger_sensors[i] = self.left_finger_motors[i].getPositionSensor()

        self.right_finger_motors[self.LEFT_FINGER] = self.getDevice("r_gripper_l_finger_joint")
        self.right_finger_motors[self.RIGHT_FINGER] = self.getDevice("r_gripper_r_finger_joint")
        self.right_finger_motors[self.LEFT_TIP] = self.getDevice("r_gripper_l_finger_tip_joint")
        self.right_finger_motors[self.RIGHT_TIP] = self.getDevice("r_gripper_r_finger_tip_joint")
    
    
        for i in range(self.RIGHT_TIP+1):
            self.right_finger_sensors[i] = self.right_finger_motors[i].getPositionSensor()

        self.head_tilt_motor = self.getDevice("head_tilt_joint")
        self.torso_motor = self.getDevice("torso_lift_joint")
        self.torso_sensor = self.getDevice("torso_lift_joint_sensor")

        self.left_finger_contact_sensors[self.LEFT_FINGER] = self.getDevice("l_gripper_l_finger_tip_contact_sensor")
        self.left_finger_contact_sensors[self.RIGHT_FINGER] = self.getDevice("l_gripper_r_finger_tip_contact_sensor")
        self.right_finger_contact_sensors[self.LEFT_FINGER] = self.getDevice("r_gripper_l_finger_tip_contact_sensor")
        self.right_finger_contact_sensors[self.RIGHT_FINGER] = self.getDevice("r_gripper_r_finger_tip_contact_sensor")

        self.imu_sensor = self.getDevice("imu_sensor")

        self.wide_stereo_l_stereo_camera_sensor = self.getDevice("wide_stereo_l_stereo_camera_sensor")
        self.wide_stereo_r_stereo_camera_sensor = self.getDevice("wide_stereo_r_stereo_camera_sensor")
        self.high_def_sensor = self.getDevice("high_def_sensor")
        self.r_forearm_cam_sensor = self.getDevice("r_forearm_cam_sensor")
        self.l_forearm_cam_sensor = self.getDevice("l_forearm_cam_sensor")
        self.laser_tilt = self.getDevice("laser_tilt")
        self.base_laser = self.getDevice("base_laser")
    
        self.head_camera = self.getDevice("wide_stereo_l_stereo_camera_sensor")
        #self.head_camera = self.getDevice("high_def_sensor")

        self.external_camera = self.getDevice("external_camera")
        self.external_camera.enable(self.TIME_STEP)

        
        #enable devices
        for i in range(8):
            self.wheel_sensors[i].enable(self.TIME_STEP)
   
        # init the motors for speed control
            self.wheel_motors[i].setPosition(float("inf"))
            self.wheel_motors[i].setVelocity(0.0)
        

        for i in range(4):
            self.rotation_sensors[i].enable(self.TIME_STEP)

        for i in range(2):
            self.left_finger_contact_sensors[i].enable(self.TIME_STEP)
            self.right_finger_contact_sensors[i].enable(self.TIME_STEP)

        for i in range(4):
            self.left_finger_sensors[i].enable(self.TIME_STEP)
            self.right_finger_sensors[i].enable(self.TIME_STEP)

        for i in range(5):
            self.left_arm_sensors[i].enable(self.TIME_STEP)
            self.right_arm_sensors[i].enable(self.TIME_STEP)

        self.torso_sensor.enable(self.TIME_STEP)
    
        self.head_camera.enable(self.TIME_STEP)
        
        
        self.set_initial_position()
        
        self.head_camera.getImage()
        self.head_camera.saveImage("img_nr_" + str(self.STEP_COUNTER) + ".jpeg",100)
        
        img = Image.open("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
        img = img.crop((0, 200, 640, 480))
        img = img.resize((320, 140), Image.ANTIALIAS) 
        img.save("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
        
        self.external_camera.getImage()
        self.external_camera.saveImage("third_person_" + "img_nr_" + str(self.STEP_COUNTER) + ".jpeg", 100)
        
        
        #self.STEP_COUNTER += 1
        pic_arr = np.array(img)
        
        
        #print(self.marker_node.getPosition())
        
        
       
        
        return pic_arr
        
    def step(self, action):
        
        #x = 2*((action[0]-(-1.20))/(0.70-(-1.20)))-1
        #y = 2*((action[1]-(-2.10))/(0.0-(-2.10)))-1
        #print(action)
        x = ((action[0]-(-1.00)) / (1.00-(-1.00))) * (0.70-(-1.20)) + (-1.20)
        y = ((action[1]-(-1.00)) / (1.00-(-1.00))) * (0.00-(-2.10)) + (-2.10)
        
        #self.set_right_arm_position(action[0], -0.01, -1.5, action[1], 1.5, True)
        self.set_right_arm_position(x, -0.01, -1.5, y, 1.5, True)

        
        """if len(self.action_list) == 0:
            x = (action[0]-(-2.29))/(0.71-(-2.29))
            y = (action[1]-(-2.32))/(0.0-(-2.32))
            action = [x, y]
            x_action = [x]
            y_action = [y]
            self.action_list = np.array([action])
            self.x_action_list = np.array([x_action])
            self.y_action_list = np.array([y_action])"""

        """else:
            #print("action")
            #print(action)
            x = (action[0]-(-2.29))/(0.71-(-2.29))
            y = (action[1]-(-2.32))/(0.0-(-2.32))
            action = [x, y]
            x_action = [x]
            y_action = [y]
            
            #print(action)
            self.action_list = np.concatenate((self.action_list, np.array([action])), axis = 0)
            self.x_action_list = np.concatenate((self.x_action_list, np.array([x_action])), axis = 0)
            self.y_action_list = np.concatenate((self.y_action_list, np.array([y_action])), axis = 0)

            """
        
        info = {}
        #reward
        reward = 0
        marker_pos = self.marker_node.getPosition()
        can_pos = self.can_node.getPosition()
        
        if (self.find_distance(marker_pos, can_pos)<= 0.15) and not ((marker_pos[2]-can_pos[2])>=abs(0.30)):
            self.correct_spot_count += 1
            reward = 1
            if self.correct_spot_count >=8:
                reward = 30
                self.done = True
        else:
            reward = 0
            self.correct_spot_count = 0   
        
        self.head_camera.getImage()
        pic_name = ("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
        self.head_camera.saveImage(pic_name, 100)
        self.external_camera.getImage()
        self.external_camera.saveImage("third_person_" + pic_name, 100)
        

        img = Image.open("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
        img = img.crop((0, 200, 640, 480))
        img = img.resize((320, 140), Image.ANTIALIAS)
        img.save("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
        
        self.state = np.array(img)
        
        self.STEP_COUNTER += 1
        #print("Correct Spot Count: " + str(self.correct_spot_count))
        
        #if reward == 1:
            #self.done = True
        if self.STEP_COUNTER >= 30:
            info["TimeLimit.truncated"] = not self.done
            self.done = True
            
        if can_pos[2] < 0.6:
            self.done = True 
        #print("Current Reward: " + str(reward))
        return self.state, reward, self.done, info

    
    def find_distance(self, a, b):
        return (math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2))
    
    
    def Demonstrations(self, model):
        
        variant = 0
        demonstration_counter = 30
        l = list(range(1, demonstration_counter+1))
        random.shuffle(l)
        print("list= ")
        print(l)
        
        for i in range(demonstration_counter):
            variant = l.pop()
            print("running demonstration nr: " + str(i))
            #variant = 31
            if variant == 1:
                
                
                new_value = [1.15, 0.15, 0.741]
                self.marker_translation.setSFVec3f(new_value)
                super().step(self.TIME_STEP)
            
                self.step([-0.10, -0.10])
                self.step([-0.10, -0.30])
                self.step([-0.20, -0.50])
                self.step([-0.20, -0.70])
                self.step([-0.20, -0.90])
                self.step([-0.20, -1.00])
                self.step([-0.20, -1.10])
                self.step([-0.20, -1.20])
                self.step([-0.20, -1.30])
                self.step([-0.20, -1.40])
                self.step([-0.20, -1.50])
                self.step([-0.20, -1.55])
                self.step([-0.20, -1.55])
                self.step([-0.20, -1.55])
                self.step([-0.20, -1.55])
                self.step([-0.20, -1.55])
                self.step([-0.20, -1.55])
                self.step([-0.20, -1.55])
                
                
                self.save_images()
                
                self.remove_images(self.STEP_COUNTER)
                self.reset()




            
      
            
            if variant == 2:
                new_value = [1.30, 0.40, 0.741]
                self.marker_translation.setSFVec3f(new_value)
                super().step(self.TIME_STEP)
                #self.reset()
            
                self.step([0.10, -0.10])
                self.step([0.20, -0.15])
                self.step([0.30, -0.20])
                self.step([0.40, -0.25])
                self.step([0.50, -0.30])
                self.step([0.50, -0.35])
                self.step([0.50, -0.35])
                self.step([0.50, -0.35])        
                self.step([0.50, -0.35])
                self.step([0.50, -0.35])
                self.step([0.50, -0.35])        
                self.step([0.50, -0.35])
                self.step([0.50, -0.35])
                self.step([0.50, -0.35])        
                self.step([0.50, -0.35])
                
                
                self.save_images()
               
                self.remove_images(self.STEP_COUNTER)
                self.reset()

            
            
            
            if variant == 3:
                new_value = [1.40, -0.30, 0.741]
                self.marker_translation.setSFVec3f(new_value)
                super().step(self.TIME_STEP)
                #self.reset()
            
                self.step([-0.10, -0.10])
                self.step([-0.20, -0.30])
                self.step([-0.30, -0.50])
                self.step([-0.40, -0.60])
                self.step([-0.50, -0.70])
                self.step([-0.60, -0.80])
                self.step([-0.60, -0.90])
                self.step([-0.60, -1.00])
                self.step([-0.60, -1.05])        
                self.step([-0.60, -1.05])
                self.step([-0.60, -1.05])
                self.step([-0.60, -1.05])
                
                self.save_images()
                
                self.remove_images(self.STEP_COUNTER)

                self.reset()


            if variant == 4:
                new_value = [1.35, 0.10, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
            
                #self.step([-0.00, -0.00])
                self.step([-0.10, -0.15])
                self.step([-0.10, -0.30])
                self.step([-0.10, -0.45])
                self.step([-0.10, -0.60])
                self.step([-0.10, -0.70])
                self.step([-0.10, -0.80])
                self.step([-0.10, -0.85])
                self.step([-0.10, -0.90])        
                self.step([-0.10, -0.95])
                self.step([-0.10, -0.95])        
                self.step([-0.10, -0.95])
                self.step([-0.10, -0.95])        
                self.step([-0.10, -0.95])
                self.step([-0.10, -0.95])

            
                self.save_images()
                
                self.remove_images(self.STEP_COUNTER)
                
                self.reset()

            
            if variant == 5:
                new_value = [1.30, 0.35, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
            
                #self.step([-0.00, 0.00])
                self.step([0.10, -0.10])
                self.step([0.10, -0.20])
                self.step([0.30, -0.30])
                self.step([0.30, -0.40])
                self.step([0.35, -0.50])
                self.step([0.35, -0.55])
                self.step([0.35, -0.55])
                self.step([0.35, -0.55])        
                self.step([0.35, -0.55])
                self.step([0.35, -0.55])
                self.step([0.35, -0.55])
                self.step([0.35, -0.55])        
                self.step([0.35, -0.55])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 6:
                new_value = [1.25, 0.15, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.00, -0.20])
                self.step([-0.10, -0.40])
                self.step([-0.10, -0.60])
                self.step([-0.10, -0.80])
                self.step([-0.10, -0.90])
                self.step([-0.10, -1.00])
                self.step([-0.10, -1.10])
                self.step([-0.10, -1.15])        
                self.step([-0.10, -1.15])
                self.step([-0.10, -1.15])        
                self.step([-0.10, -1.15])
                self.step([-0.10, -1.15])
                self.step([-0.10, -1.15])
                self.step([-0.10, -1.15])
                self.step([-0.10, -1.15])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()

                
            if variant == 7:
                new_value = [1.20, -0.15, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.10, -0.20])
                self.step([-0.20, -0.40])
                self.step([-0.30, -0.60])
                self.step([-0.40, -0.80])
                self.step([-0.50, -1.00])
                self.step([-0.60, -1.20])
                self.step([-0.70, -1.30])
                self.step([-0.70, -1.40])        
                self.step([-0.70, -1.50])
                self.step([-0.70, -1.60])        
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.60])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()

                
            if variant == 8:
                new_value = [1.40, -0.15, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.10, -0.20])
                self.step([-0.20, -0.40])
                self.step([-0.30, -0.60])
                self.step([-0.40, -0.70])
                self.step([-0.50, -0.80])
                self.step([-0.50, -0.90])
                self.step([-0.50, -1.00])
                self.step([-0.50, -1.10])        
                self.step([-0.50, -1.10])
                self.step([-0.50, -1.10])        
                self.step([-0.50, -1.10])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
                
            if variant == 9:
                new_value = [1.20, -0.25, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.15, -0.20])
                self.step([-0.30, -0.40])
                self.step([-0.40, -0.60])
                self.step([-0.50, -0.80])
                self.step([-0.60, -1.00])
                self.step([-0.70, -1.20])
                self.step([-0.80, -1.30])
                self.step([-0.85, -1.40])        
                self.step([-0.90, -1.50])
                self.step([-0.90, -1.60])        
                self.step([-0.90, -1.70])
                self.step([-0.90, -1.70])
                self.step([-0.90, -1.70])
                self.step([-0.90, -1.70])
                self.step([-0.90, -1.70])
                self.step([-0.90, -1.70])
                self.step([-0.90, -1.70])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 10:
                new_value = [1.30, 0.25, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.05, -0.20])
                self.step([0.05, -0.40])
                self.step([0.05, -0.50])
                self.step([0.05, -0.60])
                self.step([0.05, -0.70])
                self.step([0.05, -0.80])
                self.step([0.05, -0.90])
                self.step([0.05, -1.00])        
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])        
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])
                self.step([0.05, -1.00])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()

                
            if variant == 11:
                new_value = [1.10, 0.18, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.10, -0.20])
                self.step([-0.15, -0.40])
                self.step([-0.20, -0.60])
                self.step([-0.20, -0.80])
                self.step([-0.20, -1.00])
                self.step([-0.20, -1.20])
                self.step([-0.20, -1.30])
                self.step([-0.20, -1.40])        
                self.step([-0.20, -1.50])
                self.step([-0.20, -1.60])        
                self.step([-0.20, -1.60])
                self.step([-0.20, -1.60])
                self.step([-0.20, -1.60])
                self.step([-0.20, -1.60])
                self.step([-0.20, -1.60])
                self.step([-0.20, -1.60])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 12:
                new_value = [1.22, 0.33, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.00, -0.20])
                self.step([0.10, -0.40])
                self.step([0.10, -0.60])
                self.step([0.10, -0.70])
                self.step([0.10, -0.80])
                self.step([0.10, -0.90])
                self.step([0.10, -1.00])
                self.step([0.10, -1.10])        
                self.step([0.10, -1.10])
                self.step([0.10, -1.10])        
                self.step([0.10, -1.10])
                self.step([0.10, -1.10])
                self.step([0.10, -1.10])
                self.step([0.10, -1.10])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 13:
                new_value = [1.38, 0.28, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.10, -0.10])
                self.step([0.10, -0.20])
                self.step([0.20, -0.30])
                self.step([0.20, -0.40])
                self.step([0.20, -0.50])
                self.step([0.20, -0.60])
                self.step([0.20, -0.65])
                self.step([0.20, -0.65])        
                self.step([0.20, -0.65])
                self.step([0.20, -0.65])        
                self.step([0.20, -0.65])
                self.step([0.20, -0.65])
                self.step([0.20, -0.65])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 14:
                new_value = [1.21, -0.28, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.20, -0.20])
                self.step([-0.40, -0.40])
                self.step([-0.60, -0.60])
                self.step([-0.70, -0.80])
                self.step([-0.85, -1.00])
                self.step([-0.90, -1.20])
                self.step([-0.90, -1.30])
                self.step([-0.90, -1.40])        
                self.step([-0.90, -1.50])
                self.step([-0.90, -1.60])        
                self.step([-0.90, -1.65])
                self.step([-0.90, -1.65])
                self.step([-0.90, -1.65])
                self.step([-0.90, -1.65])
                self.step([-0.90, -1.65])
                self.step([-0.90, -1.65])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 15:
                new_value = [1.32, -0.37, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.20, -0.20])
                self.step([-0.40, -0.40])
                self.step([-0.50, -0.60])
                self.step([-0.60, -0.80])
                self.step([-0.70, -1.00])
                self.step([-0.80, -1.10])
                self.step([-0.80, -1.15])
                self.step([-0.80, -1.20])        
                self.step([-0.80, -1.25])
                self.step([-0.80, -1.25])        
                self.step([-0.80, -1.25])
                self.step([-0.80, -1.25])
                self.step([-0.80, -1.25])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 16:
                new_value = [1.40, -0.27, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.20, -0.15])
                self.step([-0.30, -0.30])
                self.step([-0.40, -0.45])
                self.step([-0.50, -0.60])
                self.step([-0.60, -0.75])
                self.step([-0.60, -0.90])
                self.step([-0.60, -1.00])
                self.step([-0.60, -1.00])        
                self.step([-0.60, -1.00])
                self.step([-0.60, -1.00])        
                self.step([-0.60, -1.00])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 17:
                new_value = [1.12, -0.10, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.20, -0.20])
                self.step([-0.30, -0.40])
                self.step([-0.40, -0.60])
                self.step([-0.50, -0.80])
                self.step([-0.60, -1.00])
                self.step([-0.70, -1.20])
                self.step([-0.70, -1.40])
                self.step([-0.70, -1.50])        
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.70])        
                self.step([-0.70, -1.80])
                self.step([-0.70, -1.90])
                self.step([-0.70, -1.90])
                self.step([-0.70, -1.90])
                self.step([-0.70, -1.90])
                self.step([-0.70, -1.90])
                self.step([-0.70, -1.90])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 18:
                new_value = [1.27, -0.27, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.20, -0.20])
                self.step([-0.40, -0.40])
                self.step([-0.50, -0.60])
                self.step([-0.60, -0.80])
                self.step([-0.70, -1.00])
                self.step([-0.80, -1.15])
                self.step([-0.85, -1.30])
                self.step([-0.85, -1.40])        
                self.step([-0.85, -1.50])
                self.step([-0.85, -1.55])        
                self.step([-0.85, -1.55])
                self.step([-0.85, -1.55])
                self.step([-0.85, -1.55])        
                self.step([-0.85, -1.55])
                self.step([-0.85, -1.55])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 19:
                new_value = [1.35, -0.29, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.20, -0.20])
                self.step([-0.40, -0.40])
                self.step([-0.50, -0.60])
                self.step([-0.60, -0.80])
                self.step([-0.70, -1.00])
                self.step([-0.75, -1.10])
                self.step([-0.75, -1.15])
                self.step([-0.75, -1.20])        
                self.step([-0.75, -1.25])
                self.step([-0.75, -1.25])        
                self.step([-0.75, -1.25])
                self.step([-0.75, -1.25])
                self.step([-0.75, -1.25])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()

                
            if variant == 20:
                new_value = [1.32, 0.14, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.00, -0.10])
                self.step([-0.05, -0.20])
                self.step([-0.05, -0.30])
                self.step([-0.05, -0.40])
                self.step([-0.05, -0.50])
                self.step([-0.05, -0.60])
                self.step([-0.05, -0.70])
                self.step([-0.05, -0.80])        
                self.step([-0.05, -0.90])
                self.step([-0.05, -0.90])        
                self.step([-0.05, -0.90])
                self.step([-0.05, -0.90])
                self.step([-0.05, -0.90])
                self.step([-0.05, -0.90])
                self.step([-0.05, -0.90])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()

                
            if variant == 21:
                new_value = [1.50, 0.10, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.00, -0.05])
                self.step([0.05, -0.10])
                self.step([0.10, -0.15])
                self.step([0.15, -0.20])
                self.step([0.20, -0.20])
                self.step([0.20, -0.20])
                self.step([0.20, -0.20])
                self.step([0.20, -0.20])        
                self.step([0.20, -0.20])
                self.step([0.20, -0.20])        
                self.step([0.20, -0.20])
                self.step([0.20, -0.20])
                self.step([0.20, -0.20])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()

                
            if variant == 22:
                new_value = [1.24, 0.30, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.00, -0.15])
                self.step([0.05, -0.30])
                self.step([0.10, -0.45])
                self.step([0.15, -0.60])
                self.step([0.15, -0.75])
                self.step([0.15, -0.90])
                self.step([0.15, -1.00])
                self.step([0.15, -1.00])        
                self.step([0.15, -1.00])
                self.step([0.15, -1.00])        
                self.step([0.15, -1.00])
                self.step([0.15, -1.00])
                self.step([0.15, -1.00])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 23:
                new_value = [1.34, 0.38, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.10, -0.10])
                self.step([0.20, -0.20])
                self.step([0.30, -0.30])
                self.step([0.40, -0.35])
                self.step([0.45, -0.40])
                self.step([0.45, -0.40])
                self.step([0.45, -0.40])
                self.step([0.45, -0.40])        
                self.step([0.45, -0.40])
                self.step([0.45, -0.40])        
                self.step([0.45, -0.40])
                self.step([0.45, -0.40])
                self.step([0.45, -0.40])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 24:
                new_value = [1.34, -0.23, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.15, -0.20])
                self.step([-0.30, -0.40])
                self.step([-0.40, -0.60])
                self.step([-0.50, -0.80])
                self.step([-0.60, -0.90])
                self.step([-0.65, -1.00])
                self.step([-0.65, -1.10])
                self.step([-0.65, -1.20])        
                self.step([-0.65, -1.20])
                self.step([-0.65, -1.20])
                self.step([-0.65, -1.20])
                self.step([-0.65, -1.20])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 25:
                new_value = [1.42, -0.44, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.15, -0.10])
                self.step([-0.30, -0.20])
                self.step([-0.40, -0.30])
                self.step([-0.50, -0.40])
                self.step([-0.60, -0.50])
                self.step([-0.65, -0.60])
                self.step([-0.65, -0.70])
                self.step([-0.65, -0.75])        
                self.step([-0.65, -0.75])
                self.step([-0.65, -0.75])
                self.step([-0.65, -0.75])
                self.step([-0.65, -0.75])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 26:
                new_value = [1.19, -0.12, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.15, -0.20])
                self.step([-0.30, -0.40])
                self.step([-0.40, -0.60])
                self.step([-0.50, -0.80])
                self.step([-0.60, -1.00])
                self.step([-0.70, -1.20])
                self.step([-0.70, -1.40])
                self.step([-0.70, -1.50])        
                self.step([-0.70, -1.60])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                self.step([-0.70, -1.70])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 27:
                new_value = [1.13, -0.23, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.15, -0.20])
                self.step([-0.30, -0.40])
                self.step([-0.45, -0.60])
                self.step([-0.60, -0.80])
                self.step([-0.75, -1.00])
                self.step([-0.90, -1.20])
                self.step([-1.00, -1.40])
                self.step([-1.00, -1.50])        
                self.step([-1.00, -1.60])
                self.step([-1.00, -1.70])
                self.step([-1.00, -1.80])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                self.step([-1.00, -1.90])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 28:
                new_value = [1.37, -0.34, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([-0.15, -0.20])
                self.step([-0.30, -0.40])
                self.step([-0.45, -0.60])
                self.step([-0.60, -0.70])
                self.step([-0.65, -0.80])
                self.step([-0.65, -0.90])
                self.step([-0.65, -1.00])
                self.step([-0.65, -1.00])        
                self.step([-0.65, -1.00])
                self.step([-0.65, -1.00])
                self.step([-0.65, -1.00])
                self.step([-0.65, -1.00])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 29:
                new_value = [1.37, 0.31, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.10, -0.10])
                self.step([0.20, -0.20])
                self.step([0.30, -0.30])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])        
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                self.step([0.35, -0.40])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 30:
                new_value = [1.26, 0.43, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                
                #self.step([-0.00, 0.00])
                self.step([0.10, -0.10])
                self.step([0.20, -0.20])
                self.step([0.30, -0.30])
                self.step([0.40, -0.40])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])        
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                self.step([0.50, -0.45])
                
                self.save_images()
                self.remove_images(self.STEP_COUNTER)
                self.reset()
                
            if variant == 31:
                
                new_value = [1.48, -0.37, 0.741]
                self.marker_translation.setSFVec3f(new_value) 
                super().step(self.TIME_STEP)
                self.head_camera.getImage()
                self.head_camera.saveImage("img_nr_" + str(self.STEP_COUNTER) + ".jpeg",100)
        
                img = Image.open("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
                img = img.crop((0, 200, 640, 480))
                img = img.resize((320, 140), Image.ANTIALIAS) 
                img.save("img_nr_" + "starting" + ".jpeg")
        
                pic_arr = np.array(img)
                pic_arr = pic_arr/255

                pic_arr = np.expand_dims(pic_arr, axis=0)
                print("initial obs shape: ")
                print(pic_arr.shape)

                
                while not (self.done):
                
                    action = model.predict(pic_arr)
                    #action = action.numpy()
                    print("action", action)
                    x = [action[0][0][0],action[1][0][0]]
                    action = np.array(x, dtype = np.float64)
                   
                    print(action[0])
                    print(action[1])
                    
                    x1 = action[0] * (0.71 -(-2.29)) + (-2.29)
                    x2 = action[1] * (0.0 -(-2.32)) + (-2.32)
                    print("denormalized", x1, x2)
                    action = np.array((x1, x2), dtype = np.float64)
                    print(action)
                    pic_arr, rewards, self.done, _ = self.step(action)
                    #img = Image.open("third_person_img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
                    #pic_arr = np.array(img)
                    pic_arr = pic_arr/255
                    pic_arr = np.expand_dims(pic_arr, axis=0)

                    
                    
                    
                    
                    
                
                

                       
                
                
                           
        
        
        
        self.image_list = np.delete(self.image_list, 0, axis = 0)
        self.third_person_image_list = np.delete(self.third_person_image_list, 0, axis = 0)
        print("action shape is: ")
        print(self.action_list.shape)
        print("x_action shape is: ")
        print(self.x_action_list.shape)
        print("y_action shape is: ")
        print(self.y_action_list.shape)
        
        print("imgs shape is: ")
        print(self.image_list.shape)
        
        #np.savez("demonstrations", imgs = self.image_list, actions = self.action_list, x_actions = self.x_action_list, y_actions = self.y_action_list)
        #np.savez("third_person_demonstrations", imgs = self.third_person_image_list, actions = self.action_list, x_actions = self.x_action_list, y_actions = self.y_action_list)
        return
            
            

    
    
    def ALMOST_EQUAL(self, a, b):
        return ((a < b + self.TOLERANCE) and (a > b - self.TOLERANCE))
    
    #sets wheel speeds    
    def set_wheels_speeds(self, fll, flr, frl, frr, bll, blr, brl, brr):
    

        self.wheel_motors[self.FLL_WHEEL].setVelocity(fll)
        self.wheel_motors[self.FLR_WHEEL].setVelocity(flr)
        self.wheel_motors[self.FRL_WHEEL].setVelocity(frl)
        self.wheel_motors[self.FRR_WHEEL].setVelocity(frr)
        self.wheel_motors[self.BLL_WHEEL].setVelocity(bll)
        self.wheel_motors[self.BLR_WHEEL].setVelocity(blr)
        self.wheel_motors[self.BRL_WHEEL].setVelocity(brl)
        self.wheel_motors[self.BRR_WHEEL].setVelocity(brr)
    
    

        return
    
    def set_wheels_speed(self, speed):
        self.set_wheels_speeds(speed, speed, speed, speed, speed, speed, speed, speed)
        return
        
    def stop_wheels(self):
        self.set_wheels_speeds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return
        
    # enable/disable the torques on the wheels motors
    def enable_passive_wheels(self, enable):

        if (enable):
            for i in range(8):
                self.torques[i] = self.wheel_motors[i].getAvailableTorque()
                self.wheel_motors[i].setAvailableTorque(0.0)
        else:
            for i in range(8):
                self.wheel_motors[i].setAvailableTorque(self.torques[i])
    
        return
        
    # Set the rotation wheels angles.
    # If wait_on_feedback is true, the function is left when the rotational motors have reached their target positions.
    def set_rotation_wheels_angles(self, fl, fr, bl, br, wait_on_feedback):
    
        if (wait_on_feedback):
            self.stop_wheels()
            self.enable_passive_wheels(True)

        self.rotation_motors[self.FL_ROTATION].setPosition(fl)
        self.rotation_motors[self.FR_ROTATION].setPosition(fr)
        self.rotation_motors[self.BL_ROTATION].setPosition(bl)
        self.rotation_motors[self.BR_ROTATION].setPosition(br)


        if (wait_on_feedback):
            target = [fl, fr, bl, br]

            while (True):
                all_reached = True
                for i in range(4):
                    current_position = self.rotation_sensors[i].getValue()
                    if (not self.ALMOST_EQUAL(current_position, target[i])):
                        all_reached = False
                        break

                if (all_reached):
                    break
                else:
                    #self.step()
                    super().step(self.TIME_STEP)
                    
                    self.normal_step_counter += 1
                    if self.normal_step_counter > 500:
                        #self.reset()
                        break

            self.enable_passive_wheels(False)

        return
        
        
    # High level function to rotate the robot around itself of a given angle [rad]
    # Note: the angle can be negative
    def robot_rotate(self, angle):
        self.stop_wheels()
    
        self.set_rotation_wheels_angles(3.0 * (math.pi/4), (math.pi/4), -3.0 * (math.pi/4), (-math.pi/4), True)
    
    
    
    
        max_wheel_speed = 0.0
        if angle > 0:
            max_wheel_speed = self.MAX_WHEEL_SPEED
        else:
            max_wheel_speed = -self.MAX_WHEEL_SPEED
    
    
        self.set_wheels_speed(max_wheel_speed)
    
    
    
    
        initial_wheel0_position = self.wheel_sensors[self.FLL_WHEEL].getValue()
        #expected travel distance done by the wheel
        expected_travel_distance = math.fabs(angle * 0.5 * (self.WHEELS_DISTANCE + self.SUB_WHEELS_DISTANCE))


        while (True):
            wheel0_position = self.wheel_sensors[self.FLL_WHEEL].getValue()
        # travel distance done by the wheel
            wheel0_travel_distance = math.fabs(self.WHEEL_RADIUS * (wheel0_position - initial_wheel0_position))

            if (wheel0_travel_distance > expected_travel_distance):
                break

        # reduce the speed before reaching the target
            if (expected_travel_distance - wheel0_travel_distance < 0.025):
                self.set_wheels_speed(0.1 * max_wheel_speed)
        
            #self.step()
            super().step(self.TIME_STEP)
        
  

        # reset wheels
        self.set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, True)
        self.stop_wheels()
        return


    def robot_go_forward(self, distance):

        max_wheel_speed = 0.0
        if distance > 0:
            max_wheel_speed = self.MAX_WHEEL_SPEED
        else: 
            max_wheel_speed = -self.MAX_WHEEL_SPEED
        
        self.set_wheels_speed(max_wheel_speed)

        initial_wheel0_position = self.wheel_sensors[self.FLL_WHEEL].getValue()

        while (True):
            wheel0_position = self.wheel_sensors[self.FLL_WHEEL].getValue()
            # travel distance done by the wheel
            wheel0_travel_distance = math.fabs(self.WHEEL_RADIUS * (wheel0_position - initial_wheel0_position))

            if (wheel0_travel_distance > math.fabs(distance)):
                break;

            # reduce the speed before reaching the target
            if ((math.fabs(distance) - wheel0_travel_distance) < 0.025):
                self.set_wheels_speed(0.1 * max_wheel_speed)

            #self.step()
            super().step(self.TIME_STEP)


        self.stop_wheels()
        return

    # Open or close the gripper.
    # If wait_on_feedback is true, the gripper is stopped either when the target is reached,
    # or either when something has been gripped
    def set_gripper(self, left, open, torqueWhenGripping, wait_on_feedback):

        motors = [None] * 4
        motors[self.LEFT_FINGER] = None
        if left:
            motors[self.LEFT_FINGER] = self.left_finger_motors[self.LEFT_FINGER]
        else:
            motors[self.LEFT_FINGER] = self.right_finger_motors[self.LEFT_FINGER]
  
        motors[self.RIGHT_FINGER] = None
  
        if left:
            motors[self.RIGHT_FINGER] = self.left_finger_motors[self.RIGHT_FINGER]
        else:
            motors[self.RIGHT_FINGER] = self.right_finger_motors[self.RIGHT_FINGER]
      
        motors[self.LEFT_TIP] = None
        if left:
           motors[self.LEFT_TIP] = self.left_finger_motors[self.LEFT_TIP]
        else:
            motors[self.LEFT_TIP] = self.right_finger_motors[self.LEFT_TIP]
      
        motors[self.RIGHT_TIP] = None
        if left:
            motors[self.RIGHT_TIP] = self.left_finger_motors[self.RIGHT_TIP]
        else:
            motors[self.RIGHT_TIP] = self.right_finger_motors[self.RIGHT_TIP]


        sensors = [None] * 4
        sensors[self.LEFT_FINGER] = None
        if left:
            sensors[self.LEFT_FINGER] = self.left_finger_sensors[self.LEFT_FINGER]
        else:
            sensors[self.LEFT_FINGER] = self.right_finger_sensors[self.LEFT_FINGER]
  
        sensors[self.RIGHT_FINGER] = None
        if left:
            sensors[self.RIGHT_FINGER] = self.left_finger_sensors[self.RIGHT_FINGER]
        else:
            sensors[self.RIGHT_FINGER] = self.right_finger_sensors[self.RIGHT_FINGER]
  
        sensors[self.LEFT_TIP] = None
        if left:
            sensors[self.LEFT_TIP] = self.left_finger_sensors[self.LEFT_TIP]
        else:
            sensors[self.LEFT_TIP] = self.right_finger_sensors[self.LEFT_TIP]
  
        sensors[self.RIGHT_TIP] = None
        if left:
            sensors[self.RIGHT_TIP] = self.left_finger_sensors[self.RIGHT_TIP]
        else:
            sensors[self.RIGHT_TIP] = self.right_finger_sensors[self.RIGHT_TIP]



        contacts = [None] * 2
        contacts[self.LEFT_FINGER] = None
        if left:
            contacts[self.LEFT_FINGER] = self.left_finger_contact_sensors[self.LEFT_FINGER]
        else:
            contacts[self.LEFT_FINGER] = self.right_finger_contact_sensors[self.LEFT_FINGER]
  
        contacts[self.RIGHT_FINGER] = None
        if left:
            contacts[self.RIGHT_FINGER] = self.left_finger_contact_sensors[self.RIGHT_FINGER]
        else:
            contacts[self.RIGHT_FINGER] = self.right_finger_contact_sensors[self.RIGHT_FINGER]

        firstCall = True
        maxTorque = 0.0


        if (firstCall):
            maxTorque = motors[self.LEFT_FINGER].getAvailableTorque()
            firstCall = False


        for i in range (4):
            motors[i].setAvailableTorque(maxTorque)

        if (open):
            targetOpenValue = 0.5
            for i in range(4):
                motors[i].setPosition(targetOpenValue)

            if (wait_on_feedback):
                while (not self.ALMOST_EQUAL(sensors[self.LEFT_FINGER].getValue(), targetOpenValue)):
                    #self.step()
                    super().step(self.TIME_STEP)
                    
                    self.normal_step_counter += 1
                    if self.normal_step_counter > 500:
                        #self.reset()
                        break
    
        
        else:
            targetCloseValue = 0.0
            for i in range(4):
                motors[i].setPosition(targetCloseValue)

            if (wait_on_feedback):
                # wait until the 2 touch sensors are fired or the target value is reached
                while ((contacts[self.LEFT_FINGER].getValue() == 0.0) or contacts[self.RIGHT_FINGER].getValue() == 0.0 and
                not self.ALMOST_EQUAL(sensors[self.LEFT_FINGER].getValue(), targetCloseValue)):
                    #self.step()
                    super().step(self.TIME_STEP)
                    
                    self.normal_step_counter += 1
                    if self.normal_step_counter > 500:
                        #self.reset()
                        break
      
                current_position = sensors[self.LEFT_FINGER].getValue()
                for i in range(4):
                    motors[i].setAvailableTorque(torqueWhenGripping)
                    motors[i].setPosition(max(0.0, 0.95 * current_position))

        return


    # Set the right arm position (forward kinematics)
    # If wait_on_feedback is enabled, the function is left when the target is reached.
    def set_right_arm_position(self, shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift,
                                   wrist_roll, wait_on_feedback):
                                   
        self.right_arm_motors[self.SHOULDER_ROLL].setPosition(shoulder_roll)
        self.right_arm_motors[self.SHOULDER_LIFT].setPosition(shoulder_lift)
        self.right_arm_motors[self.UPPER_ARM_ROLL].setPosition(upper_arm_roll)
        self.right_arm_motors[self.ELBOW_LIFT].setPosition(elbow_lift)
        self.right_arm_motors[self.WRIST_ROLL].setPosition(wrist_roll)

        if (wait_on_feedback):
            while (not self.ALMOST_EQUAL(self.right_arm_sensors[self.SHOULDER_ROLL].getValue(), shoulder_roll) or
               not self.ALMOST_EQUAL(self.right_arm_sensors[self.SHOULDER_LIFT].getValue(), shoulder_lift) or
               not self.ALMOST_EQUAL(self.right_arm_sensors[self.UPPER_ARM_ROLL].getValue(), upper_arm_roll) or
               not self.ALMOST_EQUAL(self.right_arm_sensors[self.ELBOW_LIFT].getValue(), elbow_lift) or
               not self.ALMOST_EQUAL(self.right_arm_sensors[self.WRIST_ROLL].getValue(), wrist_roll)):
                #self.step()
                super().step(self.TIME_STEP)
                
                self.normal_step_counter += 1
                if self.normal_step_counter > 500:
                    #self.reset()
                    break
            
        
        return
        


    # Left arm
    def set_left_arm_position(self, shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift,
                                  wrist_roll, wait_on_feedback):
        self.left_arm_motors[self.SHOULDER_ROLL].setPosition(shoulder_roll)
        self.left_arm_motors[self.SHOULDER_LIFT].setPosition(shoulder_lift)
        self.left_arm_motors[self.UPPER_ARM_ROLL].setPosition(upper_arm_roll)
        self.left_arm_motors[self.ELBOW_LIFT].setPosition(elbow_lift)
        self.left_arm_motors[self.WRIST_ROLL].setPosition(wrist_roll)

        if (wait_on_feedback):
            while (not self.ALMOST_EQUAL(self.left_arm_sensors[self.SHOULDER_ROLL].getValue(), shoulder_roll) or
               not self.ALMOST_EQUAL(self.left_arm_sensors[self.SHOULDER_LIFT].getValue(), shoulder_lift) or
               not self.ALMOST_EQUAL(self.left_arm_sensors[self.UPPER_ARM_ROLL].getValue(), upper_arm_roll) or
               not self.ALMOST_EQUAL(self.left_arm_sensors[self.ELBOW_LIFT].getValue(), elbow_lift) or
               not self.ALMOST_EQUAL(self.left_arm_sensors[self.WRIST_ROLL].getValue(), wrist_roll)):
                #self.step()
                super().step(self.TIME_STEP)
                
                self.normal_step_counter += 1
                if self.normal_step_counter > 500:
                    #self.reset()
                    break


        return


    
    # Set the torso height
    # If wait_on_feedback is enabled, the function is left when the target is reached.
    def set_torso_height(self, height, wait_on_feedback):
        self.torso_motor.setPosition(height)

        if (wait_on_feedback):
            while (not self.ALMOST_EQUAL(self.torso_sensor.getValue(), height)):
                #self.step()
                super().step(self.TIME_STEP)
                
                self.normal_step_counter +=1
                if self.normal_step_counter > 500:
                    #self.reset()
                    break

        return
        
    # Convenient initial position
    def set_initial_position(self):
        self.set_left_arm_position(0.0, 1.35, 0.0, -1.4, 0.0, False)
        #self.set_right_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, False)

        self.set_gripper(False, True, 0.0, False)
        self.set_gripper(True, True, 0.0, False)

        #self.set_torso_height(0.01, True)
        self.set_torso_height(0.02, True)
        
        self.head_tilt_motor.setPosition(0.5)
        
        #self.set_left_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
        self.set_right_arm_position(0.0, -0.01, -1.5, -0.0, 1.5, True)
        
        
        self.robot_go_forward(0.60)
        self.set_gripper(False, False, 20.0, True)


        
    
        return
    
    def save_images(self):
        for i in range(self.STEP_COUNTER):
            img = Image.open("img_nr_" + str(i) + ".jpeg")
            img2 = Image.open("third_person_img_nr_" + str(i) + ".jpeg")
            
            a = np.array(img)
            a = np.expand_dims(a, axis = 0)
            a2 = np.array(img2)
            a2 = np.expand_dims(a2, axis = 0)
            
            
            
            self.image_list = np.concatenate((self.image_list, a), axis = 0)
            self.third_person_image_list = np.concatenate((self.third_person_image_list, a2), axis = 0)
        

        
        
        return
        
    def remove_images(self, amount):
        for i in range(amount):
            os.remove("img_nr_" + str(i) + ".jpeg")
            os.remove("third_person_img_nr_" + str(i) + ".jpeg")
        return
        
    
    #def create_data(self):
    
    
    
    
    def create_bc_model(self):
        image_input = Input(shape = (140, 320, 3))
        
        inLay = Lambda(lambda x: x/255, input_shape = (140, 320, 3))(image_input)
        lay1 = Conv2D(24, (5, 5), activation = "relu", strides = (2, 2), kernel_regularizer=l2(0.001))(inLay)
        lay2 = Conv2D(36, (5, 5), activation = "relu", strides = (2, 2), kernel_regularizer=l2(0.001))(lay1)
        lay3 = Conv2D(48, (5, 5), activation = "relu", strides = (2, 2), kernel_regularizer=l2(0.001))(lay2)
        lay4 = Conv2D(64, (3, 3), activation = "relu", kernel_regularizer=l2(0.001))(lay3)
        lay5 = Conv2D(64, (3, 3), activation = "relu", kernel_regularizer=l2(0.001))(lay4)
        flatten = Flatten()(lay5)
        
        dense1 = Dense(100, activation = "relu", kernel_regularizer=l2(0.001))(flatten)
        dropout = Dropout(0.3)(dense1)

        dense2 = Dense(50, activation = "relu", kernel_regularizer=l2(0.001))(dropout)
        dropout2 = Dropout(0.3)(dense2)

        dense3 = Dense(10, activation = "relu", kernel_regularizer=l2(0.001))(dropout2)
        out1 = Dense(1)(dense3)
        out2 = Dense(1)(dense3)
        
        model = Model(inputs = [image_input], outputs = [out1, out2])
        
        

        
        #BC_model = 
        return model
        
    def train_bc_model(self, model, third_person):
        
        if third_person:
            data = np.load("third_person_demonstrations.npz")
        else:
            data = np.load("demonstrations.npz")
        
        
        
        x = data["imgs"]
        actions = data["actions"]
        x_actions = data["x_actions"]
        y_actions = data["y_actions"]
        
        #print("x for training is: ", x)
        #print("y for training is: ", y)
        print("shape of x is: ", x.shape)
        print("shape of y is: ", actions.shape)

        optimizer = Adam(lr=0.0001)
        model.compile(loss = "mse", optimizer = optimizer)
        model.fit(x, [x_actions, y_actions], shuffle = False, validation_split=0.3, batch_size = 32, epochs = 30, )
        
        return model
   
   
    def run_tests(self, model):
        
        for i in range(100):
            print("iteration nr: ", i)
            episode_reward = 0
            rew = 0
            ob = self.reset()
            new_done = True
            
            #ob = np.expand_dims(ob, axis=0)

            
            #while not self.done:
            while new_done:   
                #action, _ = model.predict(ob)
                #action = [action[0],action[1]]
                action = self.action_space.sample()
                ob, rew, self.done, _ = self.step(action)
                episode_reward += rew
                
                """action = model.predict(ob)
                x = [action[0][0][0], action[1][0][0]]
                action = np.array(x, dtype = np.float64)
                ob, rew , self.done, _ = self.step(action)
                episode_reward += rew

                ob = np.expand_dims(ob, axis=0)

                print(action)"""
                
                
                """if len(self.action_list) == 0:
                    #x = (action[0]-(-2.29))/(0.71-(-2.29))
                    #y = (action[1]-(-2.32))/(0.0-(-2.32))
                    #action = [x, y]
                    x_action = [action[0]]
                    y_action = [action[1]]
                    self.action_list = np.array([action])
                    self.x_action_list = np.array([x_action])
                    self.y_action_list = np.array([y_action])

                else:
                    #print("action")
                    #print(action)
                    #x = (action[0]-(-2.29))/(0.71-(-2.29))
                    #y = (action[1]-(-2.32))/(0.0-(-2.32))
                    #action = [x, y]
                    x_action = [action[0]]
                    y_action = [action[1]]
            
                    #print(action)
                    self.action_list = np.concatenate((self.action_list, np.array([action])), axis = 0)
                    self.x_action_list = np.concatenate((self.x_action_list, np.array([x_action])), axis = 0)
                    self.y_action_list = np.concatenate((self.y_action_list, np.array([y_action])), axis = 0)
                    #print(action)"""
                
                if self.done:
                    new_done = False   
                    
               
        
            self.total_reward_list.append(episode_reward)
            
            """self.save_images()
            self.remove_images(self.STEP_COUNTER)
        
        self.image_list = np.delete(self.image_list, 0, axis = 0)
        self.third_person_image_list = np.delete(self.third_person_image_list, 0, axis = 0)
        print("action shape is: ")
        print(self.action_list.shape)
        print("x_action shape is: ")
        print(self.x_action_list.shape)
        print("y_action shape is: ")
        print(self.y_action_list.shape)
        
        print("imgs shape is: ")
        print(self.image_list.shape)
        print("third imgs shape is: ")
        print(self.third_person_image_list.shape)"""
        
        ppo_y = []
        for i in range(len(self.total_reward_list)):
            ppo_y.append((i+1))

        plt.plot(ppo_y, self.total_reward_list)
        plt.ylabel("Reward per episode")
        plt.xlabel("Total episodes")
        plt.show()
        print("Avg is: ")
        print(sum(self.total_reward_list)/len(self.total_reward_list))
        ppo_avg = sum(self.total_reward_list)/len(self.total_reward_list)
        file_to_store = open("ppo_results", "wb")
        pickle.dump (self.total_reward_list, file_to_store)
        pickle.dump (ppo_avg, file_to_store)
        

        file_to_store.close() 
        #np.savez("demonstrations", imgs = self.image_list, actions = self.action_list, x_actions = self.x_action_list, y_actions = self.y_action_list)
        #np.savez("third_person_demonstrations", imgs = self.third_person_image_list, actions = self.action_list, x_actions = self.x_action_list, y_actions = self.y_action_list)
        return
        

    
    # unfinished version of dagger     
    def run_DAgger(self, model, dagger_itr):
        ob = self.reset()
        reward_sum = 0.0
        ob_list = []
        
        for i in range(dagger_itr):
            action = model.predict(ob)
            ob, reward, done, _ = self.step(action)
            if done:
                break
            else:
                ob_list.append(ob)
            
            reward_sum += reward
            
            
        #for ob in ob_list:
         #   images_all = np.concatenate...
          #  actions_all = np.concatenate...
        
        #model.fit...
            
        
        
        return

    def test_BC(self, model):
        
        self.head_camera.getImage()
        self.head_camera.saveImage("img_nr_" + str(self.STEP_COUNTER) + ".jpeg",100)
        
        img = Image.open("img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
        img = img.crop((0, 200, 640, 480))
        img = img.resize((320, 140), Image.ANTIALIAS) 
        #img.save("img_nr_" + "starting" + ".jpeg")
        
        pic_arr = np.array(img)
        #pic_arr = pic_arr/255

        pic_arr = np.expand_dims(pic_arr, axis=0)
        print("initial obs shape: ")
        print(pic_arr.shape)

                
        while not (self.done):
                
            action = model.predict(pic_arr)
                    #action = action.numpy()
            print("action", action)
            x = [action[0][0][0],action[1][0][0]]
            action = np.array(x, dtype = np.float64)
                   
            print(action[0])
            print(action[1])
                    
            #x1 = action[0] * (0.71 -(-2.29)) + (-2.29)
            #x2 = action[1] * (0.0 -(-2.32)) + (-2.32)
            #print("denormalized", x1, x2)
            #action = np.array((x1, x2), dtype = np.float64)
            #print(action)
            #pic_arr, rewards, self.done, _ = self.step(action)
            #img = Image.open("third_person_img_nr_" + str(self.STEP_COUNTER) + ".jpeg")
            #pic_arr = np.array(img)
            #pic_arr = pic_arr/255
            #pic_arr = np.expand_dims(pic_arr, axis=0)
    
    
def main():
    
    enviroment = ImitationGymEnviroment()
    #check_env(enviroment)
    ob = enviroment.reset()
    
    #model = PPO.load("PPO10_FINAL", env = enviroment)
    #enviroment.run_tests(model)
    
    
    
    
    #enviroment.set_right_arm_position(-0.25, -0.01, -1.5, -0.0, 1.5, True)    
    #enviroment.set_right_arm_position(-0.25, -0.01, -1.5, -1.55, 1.5, True)    

    #time.sleep(2)    
    #pic_name = ("img_env" + ".jpeg")

    #enviroment.head_camera.getImage()
    
    #enviroment.head_camera.saveImage(pic_name, 100)

    
    #img = Image.open("img_env" + ".jpeg")
    #img = img.crop((0, 200, 640, 480))
    #img = img.resize((320, 140), Image.ANTIALIAS)
    #img.save("img_env" + ".jpeg")
    
    #enviroment.external_camera.getImage()
    #enviroment.external_camera.saveImage("third_person_" + pic_name, 100)
    model = None
    #model = SAC.load("SAC_model", env = enviroment)
    #model = load_model("BC_test.h5")
    enviroment.run_tests(model)

    #enviroment.test_BC(model)
    #ob = np.expand_dims(ob, axis=0)
    
    
    #ob = ob/255
    #action = model.predict(ob)
    
    #print(action)
    """while not enviroment.done:
        action = model.predict(ob)
        x = [action[0][0][0], action[1][0][0]]
        action = np.array(x, dtype = np.float64)
        ob, _ , enviroment.done, _ = enviroment.step(action)
        ob = 
        ob = np.expand_dims(ob, axis=0)

        print(action)"""
    
    
    #model.summary()
    #enviroment.Demonstrations(model)
    
    #third_person = False
    #model = enviroment.create_bc_model()
    #model = enviroment.train_bc_model(model, third_person)
    
    #third_person = True
    #model = enviroment.train_bc_model(model, third_person)
    


    #model.save("BC_test.h5")
    
    


    
    #model = SAC("CnnPolicy", enviroment, buffer_size = 20000, learning_starts = 10000, verbose = 1, tensorboard_log = './logs/' )
        
    #checkpoint_callback = CheckpointCallback(save_freq=2048, save_path='./checkpoints/',
                                         #name_prefix='SAC_rl_model', )    
    #model = PPO('CnnPolicy', enviroment, learning_rate = 0.1, n_steps=2048, n_epochs = 10, verbose=1, )
    #model = SAC.load("SAC_model", env = enviroment)
    #model.learn(total_timesteps=3e4, tb_log_name = "SAC", callback=checkpoint_callback)
    
    
    #model.save("SAC2_model")


 
    
    
    
if __name__ == '__main__':
    main()      
