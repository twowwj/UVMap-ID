import cv2
import dlib
import numpy as np

def get_face_detector():
    return dlib.get_frontal_face_detector()

def get_landmark_predictor(model_path="shape_predictor_68_face_landmarks.dat"):
    return dlib.shape_predictor(model_path)

def align_face(image_path, model_path="shape_predictor_68_face_landmarks.dat"):
    # Load the detector and predictor
    detector = get_face_detector()
    predictor = get_landmark_predictor(model_path)
    
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces found in the image.")
        return img
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Assume landmarks[36] and landmarks[45] are the left and right eye respectively
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        # Calculate the angle between the eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # Calculate the center between the two eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # Calculate the scale of the new resulting image to be the same as the original
        scale = 1
        
        # Get the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Apply the affine transformation
        tX = img.shape[1]
        tY = img.shape[0]
        aligned_face = cv2.warpAffine(img, M, (tX, tY), flags=cv2.INTER_CUBIC)
        
        return aligned_face

# Example usage
aligned_img = align_face("path_to_your_image.jpg")
cv2.imshow("Aligned Face", aligned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
