#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'xiaosong Liu'

import os
import time
import uuid
import asyncio
import aiohttp
import aiofiles
import cv2
import numpy as np

semaphore = asyncio.Semaphore(5)

async def download_images():
    dirpath = 'pic_temp'
    url = 'https://passport.lagou.com/vcode/create?from=register&refresh=1451121012510'
    filename = 'lagou-'+ str(uuid.uuid1())
    pic_save_path = os.path.join(dirpath,filename)
    headers_ = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36",
        "Referer": 'https://passport.lagou.com/'
    }

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    async with semaphore:
        async with aiohttp.ClientSession() as session:
            async with session.get(url,headers=headers_) as html:
                try:
                    pic_images = await html.read()
                except:
                    print(html.status)
                if pic_images:
                    f = await aiofiles.open(pic_save_path + '.bmp','wb')
                    await f.write(pic_images)
                   
                    return True
                else:
                    return False


def asysnc_download():

    start_time = time.time()

    loop = asyncio.get_event_loop()
    download_tasks = [download_images() for i in range(100)]
    loop.run_until_complete(asyncio.wait(download_tasks))
    loop.close()

    end_time = time.time()

    print('下载完成: 费时%.3f秒!' % (end_time - start_time))

def load_images(path):
    subpath_or_file = os.listdir(path)
    pic_name_list = [f for f in subpath_or_file if os.path.isfile(os.path.join(path,f))] #isfile()只能判断当前目录下./,所以需要加绝对路径
    img_vec = []
    for pic_name in pic_name_list:
        image = cv2.imread(os.path.join(path,pic_name))
        img_vec.append(image)

    return img_vec

def image_show(image,img_name='image'):
    cv2.namedWindow(img_name)   
    cv2.imshow(img_name, image)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()  

#图片灰度化
def image_grayscale(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return img_gray

#均值滤波
def mean_filter(image):
    img_filter = cv2.medianBlur(image,3)
    return img_filter

#二值化
def image_binarization(image,threshold=128):
    img_height = image.shape[0]
    img_width = image.shape[1]

    for i in range(img_height):
        for j in range(img_width):
            if image[i,j]< threshold:
                image[i,j] = 0
            else:
                image[i,j] = 255
    return image

#图像锐化
def image_sharp(image,flag1=0,flag2=0):
    img_height = image.shape[0]
    img_width = image.shape[1]

    img_sharp = np.zeros(image.shape,np.uint8)
    for i in range(img_height-1):
        for j in range(img_width-1):
            if flag2 == 0:
                x = abs(image[i,j+1]-image[i,j])
                y = abs(image[i+1,j]-image[i,j])
            else:
                x = abs(image[i+1,j+1]-image[i,j])
                y = abs(image[i+1,j]-image[i,j+1])
            if flag1 == 0:
                img_sharp[i,j] = max(x,y)
            else:
                img_sharp[i,j] = x+y
    return img_sharp 

#图像的开操作； 开操作 = 腐蚀->膨胀；
def image_open(image):
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))  
    #形态学操作  
    #第二个参数：要执行的形态学操作类型，这里是开操作  
    img_open =cv2.morphologyEx(binary,cv.MORPH_OPEN,kernel)

    return img_open

#图像的闭操作；闭操作 = 膨胀->腐蚀；
def image_close(image):    
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))  
    #形态学操作  
    #第二个参数：要执行的形态学操作类型，这里是开操作  
    img_close = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)  

    return img_close
        
#对完整二维码图片垂直切割成单个字母
def image_vertical_segmentation(image,padding=4,blank=3):
    img_height = image.shape[0]
    img_width = image.shape[1]
    split_x = []
    img_split_vec = []
    x = 0
    while (True):
        if x >= img_width:
            break
        if np.sum(image[:,x]==0)>padding:
            x_start = x
            x_stride = 0
            while(True): 
                x_stride += 1
                if x + x_stride == img_width:
                    break
                if(np.sum(image[:,x+x_stride]==0)<padding):
                    break

            split_x.append([x_start,x_stride])
            x += x_stride
        x += 1

    for x in split_x:
        index_start = x[0] -blank 
        index_end = x[0]+x[1] +blank
        img_split_vec.append(image[:,index_start:index_end])

    return img_split_vec
 
 #对完整二维码图片水平切割成单个字母
def image_horizontal_segmentation(image,padding=4,blank=3):
    img_height = image.shape[0]
    img_width = image.shape[1]
    y_start = 0
    y_stride = 0
    y = 0 

    while(True):
        if y >= img_height:
            break
        if np.sum(image[y,:]==0)>padding:
            y_start = y
            while(True):
                y_stride += 1
                y_end = img_height-y_stride
                if(np.sum(image[y_end,:]==0)>padding):
                    break
            break
        y += 1

    index_start = y_start - blank 
    index_end = y_end + blank
    img_single_ = image[index_start:index_end,:]

    return img_single_

#去除噪声
def image_preprocessing(image):
    img_gray = image_grayscale(image)
    #img_binarization = image_binarization(img_gray,200)
    img_mean_filter = mean_filter(img_gray)
    #image_show(img_mean_filter)
    #img_close = image_close(img_mean_filter)
    #image_show(img_close,filename)
    img_binarization = image_binarization(img_mean_filter,220)
    #print(img_mean_filter[20:26,0:100])
    #image_show(img_binarization)
    #image_imaxsharp = image_sharp(img_binarization)
    #image_iaddsharp = image_sharp(img_binarization,1)
    #image_show(image_iaddsharp,filename)
    #iAddSharp = Sharp(image,1)
    #iRMaxSharp = Sharp(image,0,1)
    #iRAddSharp = Sharp(image,1,1)
    return img_binarization

def image_save(dirpath,filename,image):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    save_dirpath_filename = os.path.join(dirpath,filename)
    cv2.imwrite(save_dirpath_filename+'.bmp',image)

    return True

def image_expansion(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
 
    left_exp = int((img_height - img_width) / 2)
    right_exp = img_height - img_width - left_exp

    image_expansion = cv2.copyMakeBorder(image,
                                         0, 0, left_exp, right_exp,
                                         cv2.BORDER_CONSTANT,
                                         value = [255, 255, 255,255])
    return image_expansion

def main():
    dirpath = 'pic_temp'
    save_dirpath = 'split_pic_temp'
    img_vec = load_images(dirpath)

    img_single_vec = []
    for image_ in img_vec[20:25]:
        img_dropnoises = image_preprocessing(image_)
        img_split_vec = image_vertical_segmentation(img_dropnoises)
        for img_split_ in img_split_vec:
            image_show(img_split_)
            if img_split_.shape[0] >= img_split_.shape[1]: #判断图片是否垂直分割出单个字母
                img_expansion = image_expansion(img_split_)
                print(img_expansion.shape)
                image_show(img_expansion)
                #filename = str(uuid.uuid1())
                #image_save(save_dirpath,filename,img_split_)
                #img_single_ = image_horizontal_segmentation(img_split_)
                #image_show(img_single_)
                #img_single_vec.append(img_single_)
            
        #for img_single_ in img_single_vec:
        #   filename = str(uuid.uuid1())
        #    image_save(save_dirpath,filename,img_single_)

    print('图片切割并保存完成！')

if __name__ == '__main__':
    main()
    #asysnc_download()