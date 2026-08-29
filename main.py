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

def blend_feature(canvas_img, src_img, feature_name):
    dst_pts = get_landmarks(canvas_img, feature_name)
    src_pts = get_landmarks(src_img, feature_name)
    
    src_x, src_y, src_w, src_h = cv2.boundingRect(src_pts)
    dst_x, dst_y, dst_w, dst_h = cv2.boundingRect(dst_pts)
    print(
        f"{feature_name.capitalize()} size: "
        f"source={src_w}x{src_h}, target={dst_w}x{dst_h})
    
    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if matrix is None:
        raise ValueError(f"Could not calculate transformation matrix for {feature_name}.")
        
    transformed_pts = cv2.transform(np.array([src_pts], dtype=np.float32), matrix)[0]
    error = np.mean(np.linalg.norm(transformed_pts - dst_pts, axis=1))
    print(f"{feature_name.capitalize()} alignment error: {error:.1f} px")
        
    h, w = canvas_img.shape[:2]
    warped_src = cv2.warpAffine(src_img, matrix, (w, h))
    
    mask = np.zeros(canvas_img.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(dst_pts)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Step 2: Dynamic mask scaling based on feature size
    feature_size = max(dst_w, dst_h)
    
    dilate_size = max(3, int(feature_size * 0.05))
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    blur_size = max(3, int(feature_size * 0.15))
    if blur_size % 2 == 0:
        blur_size += 1
        
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    x, y, w_box, h_box = cv2.boundingRect(mask)
    
    src_crop = warped_src[y:y+h_box, x:x+w_box]
    dst_crop = canvas_img[y:y+h_box, x:x+w_box]
    mask_crop = mask[y:y+h_box, x:x+w_box]
    
    clone = cv2.seamlessClone(
        src_crop,
        dst_crop,
        mask_crop,
        (w_box // 2, h_box // 2),
        cv2.NORMAL_CLONE)
    
    result = canvas_img.copy()
    result[y:y+h_box, x:x+w_box] = clone
    
    return result

def build_v4_splicer(photo_a_path, photo_b_path, photo_c_path):
    try:
        img_a = cv2.imread(photo_a_path)
        img_b = cv2.imread(photo_b_path)
        img_c = cv2.imread(photo_c_path)

        if any(img is None for img in [img_a, img_b, img_c]):
            raise FileNotFoundError("One or more images could not be loaded. Check paths.")

        canvas = img_a.copy()
        
        canvas = blend_feature(canvas, img_b, "nose")
        canvas = blend_feature(canvas, img_c, "mouth")

        cv2.imwrite("v4_enhanced_output.jpg", canvas)
        print("Splicing complete! Saved as v4_enhanced_output.jpg")
        
    except Exception as e:
        print(f"Process failed: {str(e)}")

if __name__ == "__main__":
    build_v4_splicer(
        "Photo_a.jpg",
        "Photo_b.jpg",
        "Photo_c.jpg")
