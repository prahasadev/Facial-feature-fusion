import cv2
import numpy as np
import mediapipe as mp

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

def lab_color_transfer(source, target):
    s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    for i in range(3):
        s_mean, s_std = np.mean(s_lab[:,:,i]), np.std(s_lab[:,:,i])
        t_mean, t_std = np.mean(t_lab[:,:,i]), np.std(t_lab[:,:,i])
        s_lab[:,:,i] = (s_lab[:,:,i] - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
    s_lab = np.clip(s_lab, 0, 255).astype("uint8")
    return cv2.cvtColor(s_lab, cv2.COLOR_LAB2BGR)

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
    cv2.imwrite(f"v8_{feature_name}_aligned.jpg", warped_src_crop)
    mask_crop = np.zeros((ch, cw), dtype=np.uint8)
    hull = cv2.convexHull(local_dst_pts)
    cv2.fillConvexPoly(mask_crop, hull, 255)
    mask_crop = cv2.GaussianBlur(mask_crop, (11, 11), 0)
    matched_src_crop = lab_color_transfer(warped_src_crop, dst_crop)
    cv2.imwrite(f"v8_{feature_name}_color_matched.jpg", matched_src_crop)
    mx, my, mw, mh = cv2.boundingRect(hull)
    center = (mx + mw // 2, my + mh // 2)
    clone = cv2.seamlessClone(matched_src_crop, dst_crop, mask_crop, center, cv2.NORMAL_CLONE)
    result = canvas_img.copy()
    result[dy1:dy2, dx1:dx2] = clone
    return result

def build_v8_splicer(photo_a_path, photo_b_path, photo_c_path):
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
        cv2.imwrite("v8_final.jpg", canvas)
        print("\nSplicing complete! Saved as v8_final.jpg")
    except Exception as e:
        print(f"Process failed: {str(e)}")

if __name__ == "__main__":
    build_v8_splicer("Photo_a.jpg", "Photo_b.jpg", "Photo_c.jpg")
