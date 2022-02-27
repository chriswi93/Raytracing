# Raytracing

Sample usage:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from raytracing import RaytracingEngine

# https://pixabay.com/photos/cat-kitten-pet-striped-young-1192026/
input_img_filepath = "cat.png"

sensor_width = 1024 # screen width
sensor_height = 800 # screen height
fov_degrees = 77 # viewpoint diagonal field of view in degrees
focal_length_pu = 26 # 26 mm focal length camera (pu = physical units)
img_height_pu = 210 # 210 mm image height in physical units corresponds to the page width of a DIN A4 page
draw_grid = False # draw grid
draw_surface = True # draw surface
draw_keypoints = True # draw keypoints and lines

# rotation in 3d space (in degrees)
img_rotate_x = -35
img_rotate_y = 20
img_rotate_z = 0
rotate_xyz = np.array([img_rotate_x, img_rotate_y, img_rotate_z], dtype=np.float32)

# translation/shift in 3d space from origin (in physical units)
img_translate_x = 0
img_translate_y = 0
img_translate_z = -400 # object position is set to 400 mm in front of the screen
translate_xyz = np.array([img_translate_x, img_translate_y, img_translate_z], dtype=np.float32)

# initialize viewpoint
engine = RaytracingEngine(
    sensor_width=sensor_width,
    sensor_height=sensor_height,
    fov_degrees=fov_degrees,
    focal_length_pu=focal_length_pu,
    bg_color=(0,0,0), # background color is black
    keypoints_color=(0,255,0), # keypoint color is green
    keypoints_size=25 # keypoint size is 25
)

# load image
img = cv2.imread(input_img_filepath, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# set image and initialize vertex positions in 3d space
engine.set_image(
    img=img,
    img_height_pu=img_height_pu,
    surface_downsampling=False # resize image before drawing (saves some computation time)
)

# render frame
frame = engine.render_frame(
    rotate_xyz=rotate_xyz,
    translate_xyz=engine.pu2pixel(translate_xyz), # convert physical units to pixels
    scale_xyz=engine.get_default_scale_factor(), # scale image by physical units (image_height_pu)
    draw_surface=draw_surface,
    draw_grid=draw_grid,
    draw_keypoints=draw_keypoints,
    adjust_out_of_screen=False # auto shift if vertices are out of the visible screen area
)

# draw frame
plt.figure(figsize=(12,12))
plt.imshow(frame, cmap="gray")
```
