import sys
sys.path.append('GIT/build/python') 

# from openpose import pyopenpose as op
import mediapipe as mp
import cv2
import numpy as np

path_video = r"C:\Users\vhp9486\Documents\COMP820\Assignment 2\data\vid_patellofemerol1.mp4"

def angle_between_points(a, b, c):
    ba = np.subtract(a,b)
    bc = np.subtract(c,b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
    
def find_angles_between_vectors(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = int(np.abs(radians * 180.0 / np.pi))
    return np.degrees(angle)
        
def calculate_joint_angles(keypoints):
    joint_angles = []
    pairs = [(0, 1, 8), (1, 8, 9), (8, 9, 12),\
             (9, 12, 13), (12, 13, 10), (13, 10, 11),\
             (2, 3, 4), (5, 6, 7), (10, 11, 24),\
             (11, 24, 25), (22, 23, 24), (23, 24, 25)]

    for pair in pairs:
        a, b, c = keypoints[pair[0]], keypoints[pair[1]], keypoints[pair[2]]
        a = (a.x, a.y)
        b = (b.x, b.y)
        c = (c.x, c.y)
        joint_angles.append(find_angles_between_vectors(a, b, c))
    
    return joint_angles

def calculate_differences(joint_angles1, joint_angles2):
    differences = []
    for angle1, angle2 in zip(joint_angles1, joint_angles2):
        differences.append(abs(angle1 - angle2))
   
    return differences

def draw_stick_figure(frame, keypoints, differences, threshold=25):
    # OpenPose stick figure body part pairs
    body_parts = [[0, 1], [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23], [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20], [9, 3], [9, 7], [12, 6], [12, 10]]

    # Iterate through body part pairs and draw lines and circles
    for i, pair in enumerate(body_parts):
        partA = keypoints[pair[0]]
        partB = keypoints[pair[1]]

        if partA is not None and partB is not None:
            cv2.line(frame, (int(partA[0]), int(partA[1])), (int(partB[0]), int(partB[1])), (0, 255, 0), 2)
            if differences[i] > threshold:
                cv2.circle(frame, (int(partA[0]), int(partA[1])), 4, (0, 0, 255), -1)
                cv2.circle(frame, (int(partB[0]), int(partB[1])), 4, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (int(partA[0]), int(partA[1])), 4, (0, 255, 0), -1)
                cv2.circle(frame, (int(partB[0]), int(partB[1])), 4, (0, 255, 0), -1)


def main():
    # Set up OpenPose
    #     params = dict()
    #     params["model_folder"] = "GIT/openpose/models"
    #     opWrapper = op.WrapperPython()
    #     opWrapper.configure(params)
    #     opWrapper.start()

    # Setup mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS, \
#                                   mp_drawing.DrawingSpec(color=(245,177,66), thickness=2, circle_radius=2),\
#                                   mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    # Set up video sources
    cap = cv2.VideoCapture(0)
    video = cv2.VideoCapture(path_video)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, live_frame = cap.read()
            ret_video, video_frame = video.read()

            if not ret or not ret_video:
                break

            # OpenPose: Get keypoints for both live video source and video file
#             keypoints_live = opWrapper.get_keypoints(live_frame)
#             keypoints_video = opWrapper.get_keypoints(video_frame)

            # Mediapipe: Detection of movements in image_live
            image_live = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
            image_live.flags.writeable = False
            results_live = pose.process(image_live)
            image_live.flags.writeable = True
            image_live = cv2.cvtColor(image_live, cv2.COLOR_RGB2BGR)
            
            # Mediapipe: Detection of movements in image_video
            image_video = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            image_video.flags.writeable = False
            results_video = pose.process(image_video)
            image_video.flags.writeable = True
            image_video = cv2.cvtColor(image_video, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                keypoints_live = results_live.pose_landmarks.landmark
                keypoints_video = results_video.pose_landmarks.landmark
                            
                # Render detections in image live
                mp_drawing.draw_landmarks(image_live, results_live.pose_landmarks,mp_pose.POSE_CONNECTIONS, \
                                  mp_drawing.DrawingSpec(color=(245,177,66), thickness=2, circle_radius=2),\
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
                joint_angles_live = calculate_joint_angles(keypoints_live)
                joint_angles_video = calculate_joint_angles(keypoints_video)

                differences = calculate_differences(joint_angles_live, joint_angles_video)
                overall_percentage = sum(differences) / len(differences) * 100

                # Display the overall percentage of differences
                cv2.putText(image_live,\
                        f"Overall difference: {overall_percentage:.2f}%",\
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                pass

#             # Draw stick figures
#             draw_stick_figure(video_frame, keypoints_live[0], differences)

# numpy_horizontal_concat = np.concatenate((image_live, image_video), axis=1)
# cv2.imshow("Rehabilitation Demo", numpy_horizontal_concat)

            cv2.imshow("img1", image_live)
            cv2.imshow("img2", image_video)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    video.release()q

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
