#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:05:57 2020

@author: kob51
"""
from associate_data import associate_data
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import skimage
import scipy.interpolate
from math import floor, ceil
from scipy.sparse import csr_matrix, lil_matrix
from admm import PlugPlayADMM_deblur, get_psnr

import sys
ros_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"

if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

def create_blur_visualization(kernel_size,image,A):
    height,width, _ = image.shape
    
    x = int(kernel_size/2)
    y = int(kernel_size/2)
    
    out = []
    while x < width:
        y = int(kernel_size/2)
        
        temp  = []
        while y < height:
            temp.append(create_blur_kernel(A,kernel_size,x,y,width))
        
            y += kernel_size
        temp = np.vstack(temp)
        print(temp.shape)
        
        out.append(temp)
        
        x += kernel_size
            
        
    out = np.hstack(out)
    return out
            
    
def kernel_warp(rgb_img,kernel):
    h,w,_ = rgb_img.shape
    red = rgb_img[:,:,0]
    green = rgb_img[:,:,1]
    blue = rgb_img[:,:,2] 
    
    kernel = np.flip(kernel)
    red_out = scipy.ndimage.correlate(red, kernel, mode='constant')
    green_out = scipy.ndimage.correlate(green, kernel, mode='constant')
    blue_out = scipy.ndimage.correlate(blue, kernel, mode='constant')
    
    result = np.dstack((red_out,green_out,blue_out))
    return result
    
def warp_img(rgb_img,A):
    h,w,_ = rgb_img.shape
    red = rgb_img[:,:,0]
    green = rgb_img[:,:,1]
    blue = rgb_img[:,:,2]
    
    red_out = (A @ red.flatten()).reshape(height,width)
    green_out = (A @ green.flatten()).reshape(height,width)
    blue_out = (A @ blue.flatten()).reshape(height,width)
    
    warped = np.dstack((red_out,green_out,blue_out))
    
    return warped


def create_blur_kernel(A,kernel_size,start_x,start_y,image_width):
    
    kernel = np.zeros((kernel_size,kernel_size))
    x0 = start_x - int(kernel_size/2)
    y0 = start_y - int(kernel_size/2)
    
    x = x0
    y = y0
    
    row = pixel_to_flat_index(start_x,start_y,image_width)
    count = 0
    while x < start_x + int(kernel_size/2):
        while y < start_y + int(kernel_size/2):
            # print(y-y0)
            if pixel_to_flat_index(x,y,image_width) >= A.shape[0]:
                break
            kernel[y-y0,x-x0] = A[pixel_to_flat_index(x,y,image_width),row]
            y+=1
            count+=1
        y = y0
        x+=1
    return kernel
    
def pixel_to_flat_index(x,y,num_cols):
    result = y * num_cols + x
    return result

def get_weights(x,y):
    x1 = floor(x)
    x2 = ceil(x)
    y1 = floor(y)
    y2 = ceil(y)
    A = np.array([[1, x1, y1, x1*y1],
                  [1, x1, y2, x1*y2],
                  [1, x2, y1, x2*y1],
                  [1, x2, y2, x2*y2]])
    
    
    try:
        b = (np.linalg.inv(A)).T @ np.array([[1],
                                                 [x],
                                                 [y],
                                                 [x*y]])
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            b = np.ones((4))
            print('error')
        else:
            raise
    
    
    b /= np.sum(b)

    return b.flatten(), [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]

# eq (4) in paper
def get_homography(intrinsics,R,t,N=np.array([0,0,1]),d=1):
    H = K @ (R + (1/d)*t@N.T) @ np.linalg.inv(K)
    return H

def scale_gamma(img, scale):
    threshold = 0.0031308
    img *= scale
    gamma = np.where(img < threshold, (12.92*img), ((1.055)*img**(1/2.4) - 0.055))
    
    return gamma

if __name__ == "__main__":
    K = np.array([[726.28741455078, 0, 354.6496887207],
                  [0, 726.28741455078, 186.46566772461],
                  [0,0,1]])
    
    
    stack = "camera_shake_1" # "camera_shake_1" "plant_5"
    
    imu_file = stack + "/imu.txt"
    img_dir = stack + "/rgb"
    gt_file = stack + "/groundtruth.txt"
    
    img_data_list = associate_data(img_dir,imu_file,gt_file,skip=1)
    
    
    img1 = img_data_list[6] ##      TESTED PLANT 8 and 86, CAMERA_SHAKE 6
    print(img1.image_name)
    d = 1
    
    downsample_factor = 2
    K[:2,:] /= downsample_factor
    
    image = skimage.img_as_float32(skimage.io.imread(img_dir + "/" + img1.image_name))
    image = image[::downsample_factor,::downsample_factor,:]


    height,width,_ = image.shape
    x = range(0,width)
    y = range(0,height)

    x = 50 #25 #180 #180*2    50 50 is great!!@!!
    y = 40 #75 #100 #100*2
    kernel_size = 35
    r = int(kernel_size/2)
    
    box_pts = [[x-r,x-r,x+r,x+r,x-r],[y+r,y-r,y-r,y+r,y+r]]
    plt.imshow(image,cmap='gray')
    plt.plot(box_pts[0],box_pts[1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    x_coors = np.linspace(0,width - 1,width)
    y_coors = np.linspace(0,height - 1,height)
    indices = np.meshgrid(x_coors,y_coors)
    indices = np.stack(indices)
    indices = indices.T
    indices = indices.reshape((-1,2))
    indices = indices.T
    indices = np.vstack((indices,np.ones(height*width)))
    indices = indices.astype('int')
    
    A_t = lil_matrix((height*width,height*width))

    for i in range(img1.gt_linear_pos.shape[0]):
        print("------")
        print(i)
        
        rot_mat = R.from_euler('xyz',img1.gt_angular_pos[i,:],degrees=True).as_matrix()
        
        H = get_homography(K,rot_mat,img1.gt_linear_pos[i,:])
        
        print()
        print(H)
        
        if i > 0:
            warped_indices = H @ indices
            warped_indices /= warped_indices[2,:]
            
            for j in range(warped_indices.shape[1]):
                
                orig_x = indices[0,j]
                orig_y = indices[1,j]
                row = pixel_to_flat_index(orig_x,orig_y,width)
                
                warped_x = warped_indices[0,j]
                warped_y = warped_indices[1,j]
                
                b, coords = get_weights(warped_x,warped_y)
                
                for k in range(b.size):

                    curr_x = coords[k][0]
                    curr_y = coords[k][1]
                    col = pixel_to_flat_index(curr_x,curr_y,width)

                    if col < 0 or col >=height*width:
                        continue
                    
                    A_t[col,row] += b[k]
       
    A_t = A_t.multiply(1/i)
    A_t = A_t.tocsr()

    test = create_blur_visualization(kernel_size,image,A_t)
    plt.imshow(test,cmap='gray')
    plt.imsave("kernel_grid.png",test,cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    # warped = warp_img(image.copy(),A_t)
    # plt.imshow(warped,cmap='gray')
    # plt.title("warped")
    # plt.plot(box_pts[0],box_pts[1])
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    

    # for i in range(width):
    #     if i *30 + 31/2 >= width:
    #         break
    #     for j in range(height):
    #         if j *30 + 31/2 >= height:
    #             break
    #         test_k = create_blur_kernel(A_t,31,i*30,j*30,width)
    #         plt.imshow(test_k,cmap='gray')
    #         plt.title("(" + str(i*30) + "," + str(j*30) + ")")
    #         plt.show()
            
    kernel = create_blur_kernel(A_t,kernel_size,x,y,width)
    plt.imshow(kernel,cmap='gray')
    plt.title("blur kernel")
    plt.imsave("kernel.png",kernel,cmap='gray')
    plt.show()
    
    
    lam = 0.01
    rho = 1
    gamma = 1
    max_iters = 20
    
    final = []
    rgb = True
    
    if rgb:
        
        for i in range(3):
            test = image[:,:,i]
            out = PlugPlayADMM_deblur(test,kernel,'l1',lam,rho,gamma,max_iters)
            final.append(out)
        
        out = np.dstack(final)
        psnr = get_psnr(out,image)
    
    else:
        red_chan = image[:,:,0]
        out = PlugPlayADMM_deblur(red_chan,kernel,'l1',lam,rho,gamma,max_iters)
        psnr = get_psnr(out,red_chan)
    
    
    plt.imshow(image,cmap='gray')
    plt.title("original")
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    print("psnr:",psnr)
    
    psnr_str = "%0.2f" % psnr
    plt.imshow(out,cmap='gray')
    out[out < 0] = 0
    out[out > 1] = 1
    plt.title('Deblur result (PSNR=' + psnr_str + ')')
    plt.xticks([]), plt.yticks([])
    plt.imsave("output.png",out)
    plt.show()
    
    plt.imsave("split.png",np.vstack((image,out)))