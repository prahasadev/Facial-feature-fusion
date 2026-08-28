
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

FEATURE_INDICES = {
    "eyes": [33, 133, 362, 263, 107, 336],
    "nose": [168, 19, 1, 4, 98, 327],
    "mouth": [61, 291, 0, 17]
}

def get_auto_crop(image, feature_name, padding=20):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    
    x_coords = [int(landmarks[idx].x * w) for idx in FEATURE_INDICES[feature_name]]
    y_coords = [int(landmarks[idx].y * h) for idx in FEATURE_INDICES[feature_name]]

    x_min, x_max = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
    y_min, y_max = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)

    return image[y_min:y_max, x_min:x_max]

def build_v2_splicer(photo_a_path, photo_b_path, photo_c_path):
    img_a = cv2.imread(photo_a_path)
    img_b = cv2.imread(photo_b_path)
    img_c = cv2.imread(photo_c_path)

    if img_a is None or img_b is None or img_c is None:
        return

    eyes_crop = get_auto_crop(img_a, "eyes")
    nose_crop = get_auto_crop(img_b, "nose")
    mouth_crop = get_auto_crop(img_c, "mouth")

    if eyes_crop is None or nose_crop is None or mouth_crop is None:
        return

    target_width = eyes_crop.shape[1]

    def resize_to_width(img, t_width):
        aspect_ratio = img.shape[0] / img.shape[1]
        target_height = int(t_width * aspect_ratio)
        return cv2.resize(img, (t_width, target_height))

    nose_aligned = resize_to_width(nose_crop, target_width)
    mouth_aligned = resize_to_width(mouth_crop, target_width)

    final_portrait = np.vstack((eyes_crop, nose_aligned, mouth_aligned))

    cv2.imshow("V2 Auto-Spliced Portrait", final_portrait)
    cv2.imwrite("v2_spliced_output.jpg", final_portrait)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ =="__main__":
    pass
