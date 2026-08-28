import cv2
import numpy as np
import sys

def get_manual_crop(image, window_name):
    roi = cv2.selectROI(f"Select {window_name}", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select {window_name}")
    
    x, y, w, h = roi
    
    if w == 0 or h == 0:
        sys.exit()
        
    return image[int(y):int(y+h), int(x):int(x+w)]

def build_v1_splicer(photo_a_path, photo_b_path, photo_c_path):
    img_a = cv2.imread(photo_a_path)
    img_b = cv2.imread(photo_b_path)
    img_c = cv2.imread(photo_c_path)

    if img_a is None or img_b is None or img_c is None:
        return

    eyes_crop = get_manual_crop(img_a, "Eyes")
    nose_crop = get_manual_crop(img_b, "Nose")
    mouth_crop = get_manual_crop(img_c, "Mouth")

    target_width = eyes_crop.shape[1]

    def resize_to_width(img, t_width):
        aspect_ratio = img.shape[0] / img.shape[1]
        target_height = int(t_width * aspect_ratio)
        return cv2.resize(img, (t_width, target_height))

    nose_aligned = resize_to_width(nose_crop, target_width)
    mouth_aligned = resize_to_width(mouth_crop, target_width)

    final_portrait = np.vstack((eyes_crop, nose_aligned, mouth_aligned))

    cv2.imshow("V1 Spliced Portrait", final_portrait)
    cv2.imwrite("v1_spliced_output.jpg", final_portrait)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pass
