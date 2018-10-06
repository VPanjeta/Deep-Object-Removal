import os
import numpy as np
from cv2 import *
import tensorflow as tf
from graph_mscoco import *
from glob import glob as files

_x, _y = -1, -1
size = 320
sizeBlank, image_no, isDrawn, stroke_size = 20, 0, False, 3
font = FONT_ITALIC
# ^ Global variables

def masking(image):
	# ^ Function to generate mask for input as per the stroke

    mask = (np.array(image[:,:,0]) == 0.9)
    mask = mask & (np.array(image[:,:,1]) == 0.01)
    mask = mask & (np.array(image[:,:,2]) == 0.9)
    mask = np.dstack([mask, mask, mask]);
    return (True ^ mask) * np.array(image)

  
def mouse_callback(mouse_event, x, y, flags, parameters):
	# ^ Function for drawing lines on objects to be removed

    global _x, _y, isDrawn

    if mouse_event == EVENT_LBUTTONDOWN:
        isDrawn = True
        _x, _y = x, y

    elif mouse_event == EVENT_MOUSEMOVE:
        if isDrawn:
            line(image, (_x, _y), (x, y), (0.0, 0.0, 0.0), stroke_size)
            _x, _y = x, y

    elif mouse_event == EVENT_LBUTTONUP:
        isDrawn = False
        line(image, (_x, _y), (x, y), (0.0, 0.0, 0.0), stroke_size)


images_files = []
images_files.extend(sorted(files(os.path.join('testimages/', '*.bmp'))) )
images_path = imread(images_files[image_no]) / 255.
image = images_path
# ^ Paths for image files strored in folder testimages

text_box = np.zeros((sizeBlank, 2*size + sizeBlank, 3)) + 1.
empty = np.zeros((size, size, 3))
blank = np.zeros((size, sizeBlank, 3)) + 1

namedWindow("Deep Object Removal", WINDOW_NORMAL) 
setMouseCallback('Deep Object Removal', mouse_callback)

pretrained_model = './model/pretrained_model'
# ^ Prerained model path

sess = tf.InteractiveSession()
isTraining = tf.placeholder(tf.bool)
images_placeholder = tf.placeholder(tf.float32, shape=[1, size, size, 3], name="images")
model = Model()
reconstruction_ori = model.build_reconstruction(images_placeholder, isTraining)
saver = tf.train.Saver(max_to_keep = 100) # Maximum number of checkpoints to save
saver.restore(sess, pretrained_model) # Restoring the pretrained model
# ^ Tensorflow placeholders and variables

createTrackbar('Pen Size', 'Deep Object Removal', 1, 10, lambda x: x)
# ^ Widget for pensize

filtered_image = empty
# ^ Filtered image which is initially empty

while(True):

    sub_window = np.hstack((image, blank, filtered_image[:,:,[2, 1, 0]]))
    window = np.vstack((sub_window, text_box))
    imshow('Deep Object Removal', window)
    putText(text_box, 'Image', (110, 15), font, 0.4,(0, 0, 0), 1)
    putText(text_box, 'Filtered Image', (130 + size, 15), font, 0.4,(0, 0, 0), 1)
    # ^ Windows and text
    
    key_pressed = waitKey(1) & 0xFF

    if key_pressed == 27:
    	# [Esc] key pressed
        break

    elif key_pressed == 102: 
    	# f key pressed to filter
        input_image_masked = masking(image)
        input_image_masked = input_image_masked[:,:,[2, 1, 0]]
        shape = np.array(input_image_masked).shape
        input_tensor = np.array(input_image_masked).reshape(1, shape[0], shape[1], shape[2])
        output_tensor = sess.run(
        	reconstruction_ori,
        	feed_dict={
        		images_placeholder: input_tensor, 
        		isTraining: False
        	}
        )
        filtered_image = np.array(output_tensor)[0,:,:,:].astype(float)
        imwrite(os.path.join('results', images_files[image_no][21 : 35]), ((filtered_image[:,:,[2, 1, 0]]) * 255) )
        imwrite(os.path.join('inputs', images_files[image_no][21 : 35]), ((image) * 255))

    elif key_pressed == 114: 
    	# r key pressed to reset the image
        images_path = imread(images_files[image_no]) / 255.
        image = images_path
        filtered_image = empty

    elif key_pressed == 110: 
        # n key pressed for next image
        image_no = (image_no + 1) % len(images_files)
        images_path = imread(images_files[image_no]) / 255.
        image = images_path 
        filtered_image = empty

    # Adjust pen size
    stroke_size = getTrackbarPos('Pen Size', 'Deep Object Removal')

destroyAllWindows()
