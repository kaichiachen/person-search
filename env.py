import pickle
from copy import copy, deepcopy
import numpy as np
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from contextlib import contextmanager
from keras.applications.vgg19 import VGG19
from keras.models import Model as kerasModel
from src.reinforcement import get_image_vector, get_state
from src.utils import generate_bounding_box_from_annotation
from src.metrics import follow_iou
from src.image_helper import mask_image_with_mean_background
from src.utils import save_img

class Render(object):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        cv2.destroyAllWindows()

class Env(object):
    
    action_bound = [0, 1]
    action_dim = 5
    state_dim = 4097
    
    def __init__(self):
        with open('data/pid_map_image_update.txt', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self._data = u.load()
        self.random_index = np.arange(len(self._data))
        np.random.shuffle(self.random_index)
        self.now_index = 0
        self._epoch = 0
        self._iou_thre = 0.5
        
        self.feature_map_extractor_model = VGG19(weights='imagenet')#, include_top=False, input_shape=(800,800,3))
        self.feature_map_extractor_model = kerasModel(input=self.feature_map_extractor_model.input, output=self.feature_map_extractor_model.get_layer('fc2').output)
        self.feature_map_extractor_model.summary()
        
        self.env_render = self.gen_render()
    
    def reset(self):
        
        if self.now_index >= len(self._data):
            self.random_index = np.arange(len(self._data))
            np.random.shuffle(self.random_index)
            self.now_index = 0
            self._epoch += 1
            
        now_index = self.random_index[self.now_index]    
            
        if random.random() < 0.0:
            self._same = False
            search_index = self.random_index[int(random.random()*len(self._data))]
            while search_index == now_index or search_index >= len(self._data):
                search_index = self.random_index[int(random.random()*len(self_data))]
        else:
            search_index = now_index
            self._same = True
            
        image_index = int(random.random()*len(self._data[now_index]))
        while image_index >= len(self._data[now_index]):
            image_index = int(random.random()*len(self._data[now_index]))

        target_data = self._data[now_index][image_index]
        target_image = np.array(Image.open(target_data['image']))
        bbox = target_data['boxes'][np.where(target_data['gt_pids']==now_index)[0][0]]
        self.target_image = target_image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        image_index = int(random.random()*len(self._data[search_index]))
        while image_index >= len(self._data[search_index]):
            image_index = int(random.random()*len(self._data[search_index]))
        search_data = self._data[search_index][image_index]
        self.search_image = np.array(Image.open(search_data['image']))
    
        search_iv = get_image_vector(self.search_image, self.feature_map_extractor_model)
        self.target_iv = get_image_vector(self.target_image, self.feature_map_extractor_model)
        self.state = get_state(self.target_iv, search_iv)
        
        annotation = search_data['boxes'][np.where(search_data['gt_pids']==search_index)[0][0]]#.astype(np.int32)
        self.gt_mask = generate_bounding_box_from_annotation(annotation, self.search_image.shape)
        
        if self._same:
            region_mask = np.ones([self.search_image.shape[0], self.search_image.shape[1]])
            self._last_iou = follow_iou(self.gt_mask, region_mask)
        else:
            self._last_iou = 0
        
        self.now_index += 1
        self.last_x, self.last_y = 0, 0
        self.region_image = (self.search_image.shape[0], self.search_image.shape[1])
        self.region_masks = []
        self.state = np.append(self.state, self._last_iou)
        return self._epoch, self.state
        
    def step(self, action, final_step):
        x_ratio, y_ratio, width_ratio, height_ratio, done = action
        
        if x_ratio + width_ratio >= 1 or y_ratio + height_ratio >= 1:
            return self.state, -20, False
        
        x = int(self.last_x + self.region_image[1] * x_ratio)
        y = int(self.last_y + self.region_image[0] * y_ratio)
        width = int(self.region_image[1] * width_ratio)
        height = int(self.region_image[0] * height_ratio)
        if width < 1 or height < 1:
            if self._same:
                if final_step: return self.state, -10, True
                else: return self.state, -10, False
            else:
                return self.state, -1, True
        
        self.region_image = (int(self.region_image[0]*height_ratio),int(self.region_image[1]*width_ratio))
        
        region_mask = np.zeros([self.search_image.shape[0], self.search_image.shape[1]])
        region_mask[y:y+height,x:x+width] = 1
        iou = follow_iou(self.gt_mask, region_mask)
        if iou >= self._last_iou:
            reward = -(1-iou)#iou - self._last_iou
        else:
            reward = -5*(1-iou)#self._last_iou - iou
        self.region_masks.append(region_mask)
        
        search_iv = get_image_vector(self.search_image[y:y+height,x:x+width], self.feature_map_extractor_model)
        self.state = get_state(self.target_iv, search_iv)
        self.state = np.append(self.state, iou)
        
        self._last_iou = iou
        self.last_x = x
        self.last_y = y
        
        if self._epoch < 200: thre = 0
        else: thre = self._epoch * 0.01
        
        if thre > 0.5: thre = 0.5
        if done < thre or final_step:
            if self._same:
                if iou >= self._iou_thre:
                    return self.state, -(1-iou), True
                else:
                    return self.state, -5 * (1-iou), True
            else:
                return self.state, -0.5, True
        
        if thre == 0: return self.state, reward, iou>=0.5
        else: return self.state, reward, False
    
    def gen_render(self):
        with Render() as r:
            while True:
                yield
                im = cv2.resize(self.target_image, (self.search_image.shape[1], self.search_image.shape[0]))
                image = mask_image_with_mean_background(self.gt_mask, self.search_image, [0,255,0])
                image = np.concatenate([im] + [image],axis=1)
                image = np.concatenate([image] + [mask_image_with_mean_background(region_mask, self.search_image, [255,0,0]) for region_mask in self.region_masks],axis=1)
                image = cv2.resize(image, (1600, 800))
                cv2.imshow('Frame', image[...,::-1])
                if cv2.waitKey(1) == 27: 
                    break
    
    def render(self):
        self.env_render.send(None)
        
    def save(self):
        im = cv2.resize(self.target_image, (self.search_image.shape[1], self.search_image.shape[0]))
        image = mask_image_with_mean_background(self.gt_mask, self.search_image, [0,255,0])
        image = np.concatenate([im] + [image],axis=1)
        image = np.concatenate([image] + [mask_image_with_mean_background(region_mask, self.search_image, [255,0,0]) for region_mask in self.region_masks],axis=1)
        pid = self.random_index[self.now_index-1]
        save_img('./output/imgs/%03d-%05d-%d-%.2f.jpg' % (self._epoch+1, self.now_index, pid,self._last_iou), image)
        
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

if __name__ == '__main__':
    env = Env()
    epoch, state = env.reset()
    print(state.shape)
    #nv.render()
    #print(epoch, state)
    state, reward, iou = env.step((0.1,0.2,0.5,0.5))
    print(state.shape)
    env.render()
    
        
        
        
        
