import cv2
import torch
import numpy as np
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from PIL import Image
import os 

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image




if __name__ == "__main__":
    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    ## test 
    # image_path = '/workspace/wwang/InstantID/examples/kaifu_resize.png'
    # face_image = load_image(image_path)
    # print(face_image)
    # face_image = resize_img(face_image)
    # print(face_image)
    # face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))  
    # face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face

    # # print(face_info,'xxxxxxxxxxxxxxx')

    # !!!!!! for single image
    # image_path = '/workspace/wwang/ControlNetDreamBooth-main/UV_APP/nicu_sebe/nicu.jpg'
    # new_path = '/workspace/wwang/ControlNetDreamBooth-main/UV_APP/nicu_sebe_emb'
    # face_image = load_image(image_path)
    # print(face_image)
    # face_image = resize_img(face_image)
    # face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))  
    # face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    # face_emb = face_info['embedding']
    # # save face_emb
    # new_file = 'nicu_embed.npy'
    # new_file = os.path.join(new_path, new_file)
    # print(face_emb.shape)
    # if not os.path.exists(new_file):
    #     np.save(new_file, face_emb)



    # print('sdlkjflsakdf;asjlfjsdkalfsa')
    # dataset path
    dataset_path = '/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set'
    # embedding new path
    new_path = '/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set_emb'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    celebrites_name_dir = os.listdir(dataset_path)
    print(celebrites_name_dir)
    images = []
    i = 0
    for name_dir in celebrites_name_dir:
        name_files = os.path.join(dataset_path, name_dir)
        for file in os.listdir(name_files):
            image_path = os.path.join(name_files, file)
            print(image_path)
            # jpg , jpeg, png
            if not image_path.endswith('.jpg') and not image_path.endswith('.jpeg') and not image_path.endswith('.png'):
                continue
            face_image = load_image(image_path)
            face_image = resize_img(face_image)
            face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))  
            if len(face_info) == 0:
                print('xxxxx', file, name_dir)
                continue
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            face_emb = face_info['embedding']
            # save face_emb
            new_file = file.split('.')[0] + '.npy'
            new_file = os.path.join(new_path, name_dir, new_file)
            if not os.path.exists(os.path.join(new_path, name_dir)):
                os.makedirs(os.path.join(new_path, name_dir))
            i += 1
            print(face_emb.shape)
            if not os.path.exists(new_file):
                np.save(new_file, face_emb)
    print(i)

            


