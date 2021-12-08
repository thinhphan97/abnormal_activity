import cv2
import os
import time
import torch
from torch.nn import Softmax
import numpy as np
from src.modeling import ConvLSTM3D, ViViT
from src.config import get_cfg
from albumentations import pytorch
import albumentations as A
import threading

class Demo():
    def __init__(self, cfg):
        self.cfg = cfg 
        self.model = ViViT(cfg) 
        self.totensor = pytorch.transforms.ToTensorV2() 
        self.normalize = A.Normalize() 
        self.link_video = 0 
        self.list_image = list() 
        self.label = ["abnormal", "normal"] 
        self.action = None
        self.softmax = Softmax(dim = 1)
    def read_video(self, link_video=0):
      
        self.link_video = link_video
        if self.link_video == 0:
            self.cap = cv2.VideoCapture(self.link_video, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.link_video)
        if self.cap.isOpened() == False:
            print("Error opening video stream or file")

    def show_video(self):
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == True:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def load_model(self, link_model):

        ckpt = torch.load(link_model, "cpu")
        self.model.load_state_dict(ckpt.pop('state_dict'))
        self.model.sigmoid = Softmax(dim = 1)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def __update_list_image(self, frame):
            img = frame.copy()
            img = cv2.resize(src=img, dsize=(
                self.cfg.DATA.IMG_SIZE, self.cfg.DATA.IMG_SIZE))
            img_tensor = self.normalize(image=img)["image"]
            img_tensor = self.totensor(image=img_tensor)["image"]
            # print(img_tensor.shape)
            if len(self.list_image) < self.cfg.DATA.NUM_SLICES*self.cfg.DATA.STRIDE:
                self.list_image.append(img_tensor)
            else:
                self.list_image.pop(0)
                self.list_image.append(img_tensor)
    
    def __detect_activity(self):

        if len(self.list_image) < self.cfg.DATA.NUM_SLICES*self.cfg.DATA.STRIDE:
            self.action = None
        else:
            imgs = self.list_image[::self.cfg.DATA.STRIDE]
            imgs = torch.stack(imgs)
            imgs = imgs.unsqueeze(0)
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                
            with torch.no_grad():
                prob = self.softmax(self.model(imgs))

            if torch.cuda.is_available():
                prob = prob.cpu() 

            prob = prob.detach().numpy()
            i = np.argmax(prob, axis=1)
            # if prob[:,i[0]] > 0.9:
            self.action = self.label[i[0]]

    def show_activity(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == True:
                self.__update_list_image(frame)
                self.__detect_activity()
                text = "activity: {}"
                org = (25, 25)
                # write the text on the input image
                cv2.putText(frame, text.format(
                    self.action), org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness= 1)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_video(self, link_save, name_video):
        prev_frame_time = 0
        new_frame_time = 0
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        path = os.path.join(link_save, name_video)#"output_{}.mp4".format(str(id)))
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(
            *'mp4v'), 20, (frame_width, frame_height))
            #self.cap.get(cv2.CAP_PROP_FPS)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
        
            if ret == True:
                self.__update_list_image(frame)
                self.__detect_activity()
                text = "activity: {} fps:{}"
                org = (35, 25)

                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                
                # write the text on the input image
                cv2.putText(frame, text.format(
                    self.action, str(fps)), org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale= 2, color=(0, 255, 0), thickness=2)
                out.write(frame)
                # cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    import glob
    import time
    cfg =  get_cfg()
    stream = Demo(cfg)
    stream.load_model("weights/best_vivit_224.pth")
    videos = glob.glob("dataset/data/*.mp4")#test/abnormal/
    for video in videos:
        stream.read_video(video)
        name_video = video.split('/')[-1]
        print(name_video)
        t1 = time.time()
        stream.save_video("out_video/vivit", name_video)
        print('video: {} Time: {}'.format(name_video, time.time()-t1))
    # t1 = time.time()
    # stream.read_video("dataset/data/video_test9.mp4") #test/abnormal/video_128.mp4")
    # stream.save_video("out_video/vivit","video_test9.mp4")
    # print('Time: {}'.format(time.time()-t1))
if __name__ == '__main__':
    main()
