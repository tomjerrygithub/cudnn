import numpy as np
import cv2,os
import random
 
# calculate means and std for your data
train_txt_path = './train_val.txt'
 
CNum = 10000     # pick some to cal
 
img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
 
with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    lines = [line.split()[0] for line in lines]
    random.shuffle(lines)   # shuffle to pick pictures
 
    for i in (range(len(lines))):
        img_path = os.path.join('/', lines[i].rstrip().split()[0]) 
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        
        imgs = np.concatenate((imgs, img), axis=3)
#         print(i)
 
imgs = imgs.astype(np.float32)/255.

for i in (range(3)):
    pixels = imgs[:,:,i,:].ravel()  # flatten to line
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
                  
# cv2 BGR，PIL/Skimage RGB No Need To Trans
means.reverse() # BGR --> RGB
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
