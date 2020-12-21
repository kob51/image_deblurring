#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:34:12 2020

@author: kob51
"""

import skimage.io
import os
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# given a homegenous 4x4 transformation matrix [R t; 0 1], reverse the direction of transformation
def invertH(input_H):
    H = input_H.copy()
    
    inv = np.eye(4)
    inv[:3,:3] = H[:3,:3].T
    inv[:3,-1] = -H[:3,:3].T @ H[:3,-1]          
    
    return inv


# class to store images and their associated sensor data
class ImageData:
    def __init__(self,image_num):
        # GROUND TRUTH
        self.gt_linear_pos = None
        self.gt_angular_pos = None
        self.gt_times = None
        self.image_name = "{:10.6f}".format(image_num) +".png"
        
        # TIME
        self.image_timestamp = None
        self.raw_sensor_times = None
        self.raw_sensor_times_normalized = None
        
        # LINEAR
        self.linear_accels = None
        self.linear_vels = None
        self.linear_pos = None
        
        self.g_vector = None
        
        self.linear_vel_times = None
        self.linear_vel_times_normalized = None
        self.linear_pos_times = None
        self.linear_pos_times_normalized = None
        
        # ANGULAR
        self.angular_vels = None
        self.angular_pos = None
        
        self.angular_pos_times = None
        self.angular_pos_times_normalized = None
        
        # ROTATION
        self.rotation_matrices = [] 
        
    def setGtTimes(self,time_array):
        self.gt_times = time_array
        
    def setGtSamples(self,xyz_quat_array):
        first_sample = xyz_quat_array[0,:]
        rot_world_to_1 = R.from_quat(first_sample[3:]).as_matrix()
        
        H_world_to_1 =  np.eye(4)
        H_world_to_1[:3,:3] = rot_world_to_1 
        H_world_to_1[:3,-1] = first_sample[:3]
    
        
    
    
        H_1_to_world = invertH(H_world_to_1)
        
        # take each global sample, convert it to the frame of the first sample
        linear_list = []
        angular_list = []
        
        for i in range(xyz_quat_array.shape[0]):
            xyz = xyz_quat_array[i,:3]
            quat = xyz_quat_array[i,3:]
            H_world_to_i = np.eye(4)
            rot_world_to_i = R.from_quat(quat).as_matrix()
            H_world_to_i[:3,:3] = rot_world_to_i
            H_world_to_i[:3,-1] = xyz
            
            H_1_to_i = H_1_to_world @ H_world_to_i
            linear_list.append(H_1_to_i[:3,-1])
            angular_list.append(R.from_matrix(H_1_to_i[:3,:3]).as_euler('xyz',degrees=True))
            
        linear_list = np.stack(linear_list)
        angular_list = np.stack(angular_list)
        
        self.gt_linear_pos = linear_list
        self.gt_angular_pos = angular_list
            
    ## TIME ##########################################################
    def normalizeTimes(self,timestamp_array):
        result = timestamp_array - timestamp_array[0]
        return result
    
    def setImageTime(self,ts):
        self.image_timestamp = ts
        
    ## ANGULAR #####################################################
    def setSensorTimes(self,ts_array):
        self.raw_sensor_times = ts_array
        self.raw_sensor_times_normalized = self.normalizeTimes(self.raw_sensor_times)
        
    def setAngularVels(self,vel_array):
        self.angular_vels = vel_array
        self.calcAngularPos()
    
    def calcAngularPos(self):
        self.angular_pos = np.zeros_like(self.angular_vels)
        self.angular_pos[0,:] = np.array([0,0,0])
        self.rotation_matrices.append(np.eye(3))
        
        # correct everythign to global frame to match GT
        correction = np.array([[-1,0,0],
                                [0,-1,0],
                                [0,0,1]])

        corrected_vels = correction @ self.angular_vels.T
        
        for i in range(self.angular_vels.shape[0])[1:]:
            dt = self.raw_sensor_times[i] - self.raw_sensor_times[i-1]
            rot_init_to_current = self.rotation_matrices[-1]
            rot_current_to_init = np.linalg.inv(rot_init_to_current)
            rotated_vel =  rot_current_to_init @ corrected_vels[:,i-1]
            # print(rotated_vels.shape)
            
            # Equation 16 in paper
            new_pos = rotated_vel * dt + self.angular_pos[i-1,:]
            self.angular_pos[i,:] = new_pos
            
            self.rotation_matrices.append(R.from_euler('xyz',self.angular_pos[i,:]).as_matrix())
        
        self.angular_pos *= 180 / np.pi

    ## LINEAR #####################################################
    def setLinearAccels(self,accel_array,v_prev):
        self.linear_accels = np.zeros_like(accel_array)
        # rotated_a
        
        
        
        correction = np.eye(3)
        # correction = correction @ np.array([[1,0,0],
        #                                     [0,0,1],
        #                                     [0,-1,1]])
        
        corrected_accel_array = correction @ accel_array.T
        
        correction = np.array([[-1,0,0],
                                [0,-1,0],
                                [0,0,1]])
        
        for i in range(len(self.rotation_matrices)):
            init_to_current = self.rotation_matrices[i]
            current_to_init = np.linalg.inv(init_to_current)
            # current_to_init = init_to_current
            # self.linear_accels[i,:] = current_to_init @ accel_array[i,:]
            self.linear_accels[i,:] = correction @ current_to_init @ corrected_accel_array[:,i]
            
        # Equation 25 --> assume gravity to be average of rotated accelerations
        self.g_vector = np.mean(self.linear_accels,axis=0)
        # print(self.g_vector)

        self.calcLinearPos(v_prev)
    
    def calcLinearPos(self,v_prev):
        self.linear_vels = np.zeros_like(self.linear_accels)
        self.linear_pos = np.zeros_like(self.linear_accels)
        
        g = self.g_vector
        # g = np.array([0,0,0])
        
        # self.linear_vels[0,:] = v_prev
        # print(v_prev)
        
        for i in range(self.linear_vels.shape[0])[1:]:
            x_prev = self.linear_pos[i-1,:]
            v_prev = self.linear_vels[i-1,:]
            a_prev = self.linear_accels[i-1,:]
        
            
            dt = self.raw_sensor_times[i] - self.raw_sensor_times[i-1]
            
            # Eq 20
            v = (a_prev - g) * dt + v_prev
            self.linear_vels[i,:] = v
            
            # Eq 21
            x = (0.5 * (a_prev - g) * dt**2) + (v_prev * dt) + x_prev
            
            # Eq 24
            self.linear_pos[i,:] = x
            
            
 ############################################################################################    
       
def read_data_file(filepath):
    f = open(filepath,'r')
    samples = f.read()
    samples = samples.split('\n')
    data_format = samples[0]
    samples = samples[1:]
    samples = [x for x in samples if x != '']
    samples = [np.array(x.split(),dtype='float') for x in samples]
    samples = np.stack(samples)
    f.close()
    
    return samples, data_format

def associate_data(img_dir,imu_file,gt_file,skip=1):
    # get filenames, strip off numbers
    image_names = sorted(os.listdir(img_dir))
    image_nums = [float(x[:-4]) for x in image_names]
    
    # read imu file, format into (num_samples) x 7 array
    # Format: timestamp angular_velocity[rad/sec](x y z) linear_acceleration[m/s^2](x y z)'
    
    imu_samples,imu_format = read_data_file(imu_file)
    
    gt_samples,gt_format = read_data_file(gt_file)
    print(gt_format)
    
    # associate imu and gt readings with images
    j = 0
    k = 0
    image_data_list = []
    gt_readings_per_image = []


    v_prev = np.array([0,0,0])

    for i in range(len(image_nums))[:-1:skip]: 
        # print('here')
        im_data = ImageData(image_nums[i])
        im_data.setImageTime(image_nums[i])
        temp_imu = []
        while imu_samples[j,0] <= image_nums[i]:
            temp_imu.append(imu_samples[j,:])
            j += 1
            if j == imu_samples.shape[0]:
                break
            
        temp_imu = np.stack(temp_imu).astype('float')
        im_data.setSensorTimes(temp_imu[:,0])
        im_data.setAngularVels(temp_imu[:,1:4])
        im_data.setLinearAccels(temp_imu[:,4:],v_prev)
            
        temp_gt = []
        while gt_samples[k,0] <= image_nums[i]:
            temp_gt.append(gt_samples[k,:])
            k+=1
            if k == gt_samples.shape[0]:
                break
        
        
        temp_gt = np.stack(temp_gt).astype('float')
        im_data.setGtTimes(temp_gt[:,0])
        im_data.setGtSamples(temp_gt[:,1:])
        
        v_prev = im_data.linear_vels[-1,:]
        
        image_data_list.append(im_data)
        
    return image_data_list

if __name__ == "__main__":
    
    stack_name = "plant_5" #"test" # "plant_5" # "plant_5" # camera_shake_1
    
    imu_file = stack_name + "_imu/imu.txt"
    img_dir = stack_name + "/rgb"
    gt_file = stack_name + "/groundtruth.txt"
    
    image_data_list = associate_data(img_dir,imu_file,gt_file,1)
    
    single = False
    if single:
        
        a = image_data_list[4]
        
        x_pos  = a.linear_pos[:,0]
        y_pos = a.linear_pos[:,1]
        z_pos = a.linear_pos[:,2]
        
        gt_x = a.gt_linear_pos[:,0]
        gt_y = a.gt_linear_pos[:,1]
        gt_z = a.gt_linear_pos[:,2]
        
        roll = a.angular_pos[:,0]
        pitch = a.angular_pos[:,1]
        yaw = a.angular_pos[:,2]
        
        gt_roll = a.gt_angular_pos[:,0]
        gt_pitch = a.gt_angular_pos[:,1]
        gt_yaw = a.gt_angular_pos[:,2]
        
        plt.title("GT")
        plt.ylabel('Position (m)')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.plot(a.gt_times - a.gt_times[0],gt_x,label='gt_x')
        plt.plot(a.gt_times - a.gt_times[0],gt_y,label='gt_y')
        plt.plot(a.gt_times - a.gt_times[0],gt_z,label='gt_z')
        plt.show()
        
        ds_factor = 1
        
        plt.title("MEASURED")
        plt.plot((a.raw_sensor_times - a.raw_sensor_times[0])[::ds_factor],x_pos[::ds_factor],label='x')
        plt.plot((a.raw_sensor_times  - a.raw_sensor_times[0])[::ds_factor],y_pos[::ds_factor],label='y')
        plt.plot((a.raw_sensor_times  - a.raw_sensor_times[0])[::ds_factor],z_pos[::ds_factor],label='z')
        plt.legend()
        plt.show()
        
    else:
        plot = False
        for i in range(len(image_data_list)):
            print('--------------------')
            print(image_data_list[i].image_name, i+1,"/",len(image_data_list))
            a = image_data_list[i]
            
            x_pos  = a.linear_pos[:,0]
            y_pos = a.linear_pos[:,1]
            z_pos = a.linear_pos[:,2]
            
            gt_x = a.gt_linear_pos[:,0]
            gt_y = a.gt_linear_pos[:,1]
            gt_z = a.gt_linear_pos[:,2]
            
            roll = a.angular_pos[:,0]
            pitch = a.angular_pos[:,1]
            yaw = a.angular_pos[:,2]
            
            gt_roll = a.gt_angular_pos[:,0]
            gt_pitch = a.gt_angular_pos[:,1]
            gt_yaw = a.gt_angular_pos[:,2]
            
            if plot:
                fig = plt.figure()
                fig.suptitle("Linear Positions")
                ax = fig.add_subplot(2,1,1)
                
                ax.set_title("Measured")
                plt.plot(a.raw_sensor_times - a.raw_sensor_times[0],x_pos,label='x')
                plt.plot(a.raw_sensor_times  - a.raw_sensor_times[0],y_pos,label='y')
                plt.plot(a.raw_sensor_times  - a.raw_sensor_times[0],z_pos,label='z')
                
                plt.ylabel('Position (m)')
                plt.legend()
                ax = fig.add_subplot(2,1,2)
                ax.set_title("GT")
                plt.xlabel('Time (s)')
                plt.ylabel('Position (m)')
                plt.plot(a.gt_times - a.gt_times[0],gt_x,label='gt_x')
                plt.plot(a.gt_times - a.gt_times[0],gt_y,label='gt_y')
                plt.plot(a.gt_times - a.gt_times[0],gt_z,label='gt_z')
                plt.legend()
                plt.legend()
                plt.show()
            
                
            
                fig = plt.figure()
                fig.suptitle("Angular Positions")
                ax = fig.add_subplot(2,1,1)
                
                ax.set_title("Measured")
                plt.plot(a.raw_sensor_times - a.raw_sensor_times[0],roll,label='roll')
                plt.plot(a.raw_sensor_times  - a.raw_sensor_times[0],pitch,label='pitch')
                plt.plot(a.raw_sensor_times  - a.raw_sensor_times[0],yaw,label='yaw')
                
                plt.ylabel('Position (deg)')
                plt.legend()
                ax = fig.add_subplot(2,1,2)
                ax.set_title("GT")
                plt.xlabel('Time (s)')
                plt.ylabel('Position (deg)')
                plt.plot(a.gt_times - a.gt_times[0],gt_roll,label='gt_roll')
                plt.plot(a.gt_times - a.gt_times[0],gt_pitch,label='gt_pitch')
                plt.plot(a.gt_times - a.gt_times[0],gt_yaw,label='gt_yaw')
                plt.legend()
                plt.legend()
                plt.show()
        
    
        
        