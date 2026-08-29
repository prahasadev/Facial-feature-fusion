import cv2
import numpy as np
import mediapipe as mp
from skimage import exposure

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

FEATURE_INDICES = {
    "nose": [168, 275, 330, 327, 326, 2, 97, 98, 101, 45],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]}

def get_landmarks(image, feature_name):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        raise ValueError(f"Face not detected for {feature_name}.")
    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in FEATURE_INDICES[feature_name]]
    return np.array(pts, dtype=np.int32)

def blend_feature(canvas_img, src_img, feature_name):
    dst_pts = get_landmarks(canvas_img, feature_name)
    src_pts = get_landmarks(src_img, feature_name)
    dx, dy, dw, dh = cv2.boundingRect(dst_pts)
    dst_pad = 30
    h_dst, w_dst = canvas_img.shape[:2]
    dx1, dy1 = max(0, dx - dst_pad), max(0, dy - dst_pad)
    dx2, dy2 = min(w_dst, dx + dw + dst_pad), min(h_dst, dy + dh + dst_pad)
    dst_crop = canvas_img[dy1:dy2, dx1:dx2].copy()
    local_dst_pts = dst_pts - np.array([dx1, dy1], dtype=np.int32)
    sx, sy, sw, sh = cv2.boundingRect(src_pts)
    src_pad = max(sw, sh)
    h_src, w_src = src_img.shape[:2]
    sx1, sy1 = max(0, sx - src_pad), max(0, sy - src_pad)
    sx2, sy2 = min(w_src, sx + sw + src_pad), min(h_src, sy + sh + src_pad)
    src_crop = src_img[sy1:sy2, sx1:sx2].copy()
    local_src_pts = src_pts - np.array([sx1, sy1], dtype=np.int32)
    matrix, _ = cv2.estimateAffinePartial2D(local_src_pts, local_dst_pts)
    if matrix is None:
        raise ValueError(f"Could not calculate local transformation matrix for {feature_name}.")
    transformed_pts = cv2.transform(np.array([local_src_pts], dtype=np.float32), matrix)[0]
    error = np.mean(np.linalg.norm(transformed_pts - local_dst_pts, axis=1))
    print(f"{feature_name.capitalize()} local alignment error: {error:.1f} px")
    cw, ch = (dx2 - dx1), (dy2 - dy1)
    warped_src_crop = cv2.warpAffine(src_crop, matrix, (cw, ch))
    cv2.imwrite(f"v5_{feature_name}_aligned.jpg", warped_src_crop)
    mask_crop = np.zeros((ch, cw), dtype=np.uint8)
    hull = cv2.convexHull(local_dst_pts)
    cv2.fillConvexPoly(mask_crop, hull, 255)
    if feature_name == "mouth":
        mask_crop = cv2.dilate(mask_crop, np.ones((9, 9), np.uint8), iterations=1)
        mask_crop = cv2.GaussianBlur(mask_crop, (15, 15), 0)
    else:
        mask_crop = cv2.GaussianBlur(mask_crop, (11, 11), 0)
    matched_src_crop = exposure.match_histograms(warped_src_crop, dst_crop, channel_axis=-1)
    mx, my, mw, mh = cv2.boundingRect(hull)
    center = (mx + mw // 2, my + mh // 2)
    clone = cv2.seamlessClone(matched_src_crop, dst_crop, mask_crop, center, cv2.NORMAL_CLONE)
    result = canvas_img.copy()
    result[dy1:dy2, dx1:dx2] = clone
    return result

def build_v5_splicer(photo_a_path, photo_b_path, photo_c_path):
    try:
        img_a = cv2.imread(photo_a_path)
        img_b = cv2.imread(photo_b_path)
        img_c = cv2.imread(photo_c_path)
        if any(img is None for img in [img_a, img_b, img_c]):
            raise FileNotFoundError("One or more images could not be loaded. Check paths.")
        canvas = img_a.copy()
        print("--- Blending Nose ---")
        canvas = blend_feature(canvas, img_b, "nose")
        print("\n--- Blending Mouth ---")
        canvas = blend_feature(canvas, img_c, "mouth")
        cv2.imwrite("v5_final.jpg", canvas)
        print("\nSplicing complete! Check v5_final.jpg and debug files.")
    except Exception as e:
        print(f"Process failed: {str(e)}")

if __name__ == "__main__":
    build_v5_splicer("Photo_a.jpg", "Photo_b.jpg", "Photo_c.jpg")
