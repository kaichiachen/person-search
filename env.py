import pickle
from copy import copy, deepcopy
import tensorflow as tf
import numpy as np
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from contextlib import contextmanager
from keras.applications.vgg19 import VGG19
from keras.models import Model as kerasModel
import keras.backend.tensorflow_backend as KTF
from src.reinforcement import get_image_vector, get_state
from src.utils import generate_bounding_box_from_annotation
from src.metrics import follow_iou
from src.image_helper import mask_image_with_mean_background
from src.utils import save_img
from src.features import roi, get_roi_model

class Render(object):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        cv2.destroyAllWindows()

class Env(object):
    
    action_bound = [0, 1]
    action_dim = 5
    state_dim = 1000 * 2 + 4 * 5
    
    def __init__(self):

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #sess = tf.Session(config=tf.ConfigProto(device_count={'cpu':2}))
        self.sess = tf.Session()
        KTF.set_session(self.sess)
        with open('data/pid_map_image_update.txt', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self._data = u.load()
        self._data = self._data[:100]
        self.random_index = np.arange(len(self._data))
        np.random.shuffle(self.random_index)
        self.now_index = 0
        self._epoch = 0
        self._iou_thre = 0.5
        
        self.feature_map_extractor_model = VGG19(weights='imagenet' ,include_top=False)#, input_shape=(300,300,3))
        self.v = VGG19(weights='imagenet')
        self.roi_model = get_roi_model(self.v)
        #self.v = kerasModel(inputs=self.v.input, outputs=self.v.get_layer('fc2').output)
        
        self.feature_map_extractor_model.summary()
        self.roi_model.summary()
        
        self.env_render = self.gen_render()
    
    def reset(self, max_step):
        self.max_step = max_step
        self.history_action = np.zeros(4 * 5)
        self.history_state = np.zeros(4096 * max_step)
        
        if self.now_index >= len(self._data):
            self.random_index = np.arange(len(self._data))
            np.random.shuffle(self.random_index)
            self.now_index = 0
            self._epoch += 1
            np.random.shuffle(self.random_index)
            
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

        #target_data = self._data[3000][0]
        target_data = self._data[now_index][image_index]
        target_image = np.array(Image.open(target_data['image']))
        #bbox = target_data['boxes'][np.where(target_data['gt_pids']==3000)[0][0]]
        bbox = target_data['boxes'][np.where(target_data['gt_pids']==now_index)[0][0]]
        self.target_image = target_image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        image_index = int(random.random()*len(self._data[search_index]))
        while image_index >= len(self._data[search_index]):
            image_index = int(random.random()*len(self._data[search_index]))
        #search_data = self._data[3000][1]
        search_data = self._data[search_index][image_index]
        self.search_image = np.array(Image.open(search_data['image']))
        ori_search_image_shape = self.search_image.shape
        self.search_image = cv2.resize(self.search_image, (224, 224)).astype(np.float32)
        search_iv = get_image_vector(self.search_image, self.v)
        self.features = get_image_vector(self.search_image, self.feature_map_extractor_model)
        #self.target_iv = get_image_vector(self.target_image, self.feature_map_extractor_model)
        self.target_iv = get_image_vector(self.target_image, self.v)
        
        #self.history_state[:4096] = search_iv
        self.history_action[:4] = 0,0,1,1
        self.state = get_state(self.target_iv, search_iv, self.history_action)
        
        self.annotation = search_data['boxes'][np.where(search_data['gt_pids']==now_index)[0][0]]
        self.annotation = [int(self.annotation[0]/ori_search_image_shape[1]*self.search_image.shape[1]),
                      int(self.annotation[1]/ori_search_image_shape[0]*self.search_image.shape[0]),
                      int(self.annotation[2]/ori_search_image_shape[1]*self.search_image.shape[1]),
                      int(self.annotation[3]/ori_search_image_shape[0]*self.search_image.shape[0])]
        
        #annotation = search_data['boxes'][np.where(search_data['gt_pids']==search_index)[0][0]]#.astype(np.int32)
        self.gt_mask = generate_bounding_box_from_annotation(self.annotation, self.search_image.shape)
        
        if self._same:
            region_mask = np.ones([self.search_image.shape[0], self.search_image.shape[1]])
            self.init_iou = self._last_iou = follow_iou(self.gt_mask, region_mask)
        else:
            self.init_iou = self._last_iou = 0
        
        self.now_index += 1
        self.last_x, self.last_y = 0, 0
        self.region_image = (self.search_image.shape[1], self.search_image.shape[0])
        self.region_masks = []
        return self._epoch, self.state
        
    def step(self, action, s):
        info = {
        }
        final_step = (s == self.max_step)
        x,y,xx,yy,done = self._do_continuous_action(action)
        #print(self.annotation,x,y,xx,yy)
        width = xx-x
        height = yy-y
        info['x']=x
        info['y']=y
        info['width']=width
        info['height']=height
        info['iou'] = self._last_iou
        
        if width < 1 or height < 1:
            info['iou'] = self._last_iou
            if self._same:
                if final_step: return self.state, -10, True, info
                else: return self.state, -50, True, info
            else:
                return self.state, 0.5, True, info
        
        region_mask = np.zeros([self.search_image.shape[0], self.search_image.shape[1]])
        region_mask[y:y+height,x:x+width] = 1
        iou = follow_iou(self.gt_mask, region_mask)
        info['iou'] = iou
#         if iou > self._last_iou:
#             info['iou'] = iou
#         else:
#             info['iou']=self._last_iou
        
        if iou > self._last_iou:
            reward = -0.5 + iou - self._iou_thre
#             if iou >= self._iou_thre:
#                 reward = 10
        else:
            reward = -0.5 + iou - self._iou_thre
            #return self.state, reward*5, False, info
        
        if final_step and not self._same:
            return self.state, 0.5, True, info
        
        if done:
            if not self._same:
                return self.state, 0.5, True, info
            if iou >= self._iou_thre:
                return self.state, 10, True, info
            else:
                return self.state, reward*(self.max_step-s+1)*10, True, info
        
        self.region_image = (width,height)
        
        self.region_masks.append(region_mask)
        
        self._last_iou = iou
        self.last_x = x
        self.last_y = y
        
        #search_iv = get_image_vector(self.search_image[y:y+height,x:x+width], self.feature_map_extractor_model)
        X_roi = np.array([[x,y,x+width,y+height]])
        X_roi = np.reshape(X_roi, (1, 1, 4))
        search_iv = self.roi_model.predict([self.features, X_roi])[0]
        self.history_action[(s%5)*4:((s%5)+1)*4] = action[:4]
        self.state = get_state(self.target_iv, search_iv, self.history_action)
        
        return self.state, reward, False, info
    
    def continuous_action_knowledge(self):
        x,y,xx,yy = tuple(self.annotation)
        x_ratio = (x-min(self.last_x,x)) / self.region_image[0]
        y_ratio = (y-min(self.last_y,y)) / self.region_image[1]
        xx_ratio = (xx-min(self.last_x,xx)) / self.region_image[0]
        yy_ratio = (yy-min(self.last_y,yy)) / self.region_image[1]
        
        return random.uniform(0, x_ratio),random.uniform(0, y_ratio),random.uniform(1, xx_ratio),random.uniform(1, yy_ratio), 0.7
        
#         new_x=int(random.uniform(min(self.last_x,x),x))
#         if x-min(self.last_x,x)!=0:
#             new_x_ratio = (new_x-min(self.last_x,x))/(x-min(self.last_x,x))
#         else:
#             new_x_ratio=0
#         new_y=int(random.uniform(min(self.last_y,y),y))
#         if y-min(self.last_y,y)!=0:
#             new_y_ratio = (new_y-min(self.last_y,y))/(y-min(self.last_y,y))
#         else:
#             new_y_ratio=0
#         new_xx=int(random.uniform(xx,max(xx,self.region_image[0])))
#         new_yy=int(random.uniform(yy,max(yy,self.region_image[1])))
        
    def _do_continuous_action(self, action):
        x_ratio, y_ratio, xx_ratio, yy_ratio, done = action
        
#         x_ratio /= 2
#         y_ratio /= 2
#         width_ratio /= 2
#         height_ratio /= 2
#         width_ratio = 0.5 + width_ratio - x_ratio
#         height_ratio = 0.5 + height_ratio - y_ratio
        
#         x = int(self.last_x + self.region_image[1] * x_ratio)
#         y = int(self.last_y + self.region_image[0] * y_ratio)
#         width = int(self.region_image[0] * width_ratio)
#         height = int(self.region_image[1] * height_ratio)
        
        x = int(self.last_x + self.region_image[0] * x_ratio)
        y = int(self.last_y + self.region_image[1] * y_ratio)
        xx = int(self.last_x + self.region_image[0] * xx_ratio)
        yy = int(self.last_y + self.region_image[1] * yy_ratio)
        return x,y,xx,yy,done > 0.8
    
    def _do_discrete_action(self, action):
        pass
    
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
        
    def get_img(self):
        im = cv2.resize(self.target_image, (self.search_image.shape[1], self.search_image.shape[0]))
        image = mask_image_with_mean_background(self.gt_mask, self.search_image, [0,255,0])
        image = np.concatenate([im] + [image],axis=1)
        image = np.concatenate([image] + [mask_image_with_mean_background(region_mask, self.search_image, [255,0,0]) for region_mask in self.region_masks],axis=1)
        return image
        
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
    
        
        
        
        
