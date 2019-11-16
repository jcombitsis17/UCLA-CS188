import cv2
from skimage.feature import match_template
from skimage import data
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt


video_file = 'test1.mov'
template_image = 'football.png'

def play_video():
    video = cv2.VideoCapture(video_file)

    # Read until video is completed
    while(video.isOpened()):
        ret, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Frame',frame_gray)
    
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
            
    video.release()
    cv2.destroyAllWindows()

def match_templates(play=False):
    template = imread(template_image,as_gray='True')
    height, width = template.shape

    video = cv2.VideoCapture(video_file)

    frame_count = 1
    matched = []
    frames = []

    x_coord = []
    y_coord = []
    print("Running template matching")
    while(video.isOpened()):
        ret, frame = video.read()
        if frame_count % 25 == 0 or frame_count == 207:
            print("{}/207".format(frame_count))
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            result = match_template(frame_gray,template,pad_input=True)
            matched.append(result)
            frames.append(frame)
            frame_count += 1

        else:
            break
            
    print("Playing result")
    for i in range(len(matched)):
        result = matched[i]
        frame = frames[i]
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]

        x_coord.append(x)
        y_coord.append(y)
        
        if play==True:
            #display location of matched template on frame
            rect = cv2.rectangle(frame, (x-width//2, y-height//2), (x+width//2, y+height//2), (0,0,255), 10)
            rect = cv2.putText(rect,'Frame: {}'.format(i), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            cv2.imshow('Frame', rect)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    

    video.release()
    cv2.destroyAllWindows()



    # shows the cross correlation
    # plt.figure()
    # plt.imshow(result, cmap='gray')
    # plt.xlabel('Pixel location in X Direction')
    # plt.ylabel('Pixel location in Y Direction')
    # plt.colorbar()

    # plt.show()

    # plt.figure()
    # plt.plot(x_coord,y_coord)
    # plt.show()

    return x_coord, y_coord, frames


def synthetic_focus(shiftx, shifty, images):
    shifted_frames = []
    ref_x = shiftx[0]
    ref_y = shifty[0]

    res = np.zeros(images[0].shape)

    print('Adding Images...')
    for i,image in enumerate(images):
        translation = np.float32([[1,0,ref_x-shiftx[i]],[0,1,ref_y-shifty[i]]])
        shifted = cv2.warpAffine(image, translation, (image.shape[1], image.shape[0]))
        shifted_frames.append(shifted)
        res += shifted / len(images)
    print('Done.')
    
    focused_frame = res.astype(np.uint8)
    # focused_frame = np.sum(shifted_frames, axis=0)
    # normalized_frame = np.zeros(shape=(1080, 1920))
    # normalized_frame = cv2.normalize(focused_frame, normalized_frame, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('Focused Frame',focused_frame)
    while not cv2.waitKey(0) & 0xFF == ord('q'):
        pass

if __name__ == '__main__':
    #play_video()
    x, y, frames = match_templates(False)
    synthetic_focus(x, y, frames)