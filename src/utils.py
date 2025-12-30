import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):
    """
    Overlap a image 
    """
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background

    
    bg_x = max(0, x)
    bg_y = max(0, y)
    bg_w_slice = min(bg_w, x + ov_w) - bg_x
    bg_h_slice = min(bg_h, y + ov_h) - bg_y

    ov_x = max(0, -x)
    ov_y = max(0, -y)
    
    
    overlay_crop = overlay[ov_y:ov_y+bg_h_slice, ov_x:ov_x+bg_w_slice]
    background_crop = background[bg_y:bg_y+bg_h_slice, bg_x:bg_x+bg_w_slice]

    if overlay_crop.shape[2] < 4:
        background[bg_y:bg_y+bg_h_slice, bg_x:bg_x+bg_w_slice] = overlay_crop
        return background

    alpha_channel = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_channel

    
    for c in range(3):
        background_crop[:, :, c] = (alpha_channel * overlay_crop[:, :, c] + 
                                    alpha_inv * background_crop[:, :, c])

    return background