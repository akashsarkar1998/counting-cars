import cv2
import os
import numpy as np

def find_class_id(classes_file, class_label):
    with open(classes_file, 'r') as file:
        classes = file.read().strip().split('\n')
    return classes.index(class_label)

def count_cars(video_path):
    # Get the absolute path to the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to YOLO model files using raw string literals
    weights_path = os.path.join(script_dir, r'D:\1. APPS\Works\Akash1\Projects\car\yolov3.weights')
    cfg_path = os.path.join(script_dir, r'D:\1. APPS\Works\Akash1\Projects\car\yolov3.cfg')
    classes_file = os.path.join(script_dir, r'D:\1. APPS\Works\Akash1\Projects\car\coco.names')  # Adjust this path

    # Initialize YOLO object detector
    net = cv2.dnn.readNet(weights_path, cfg_path)

    # Initialize MIL tracker (you can try other available trackers)
    tracker = cv2.TrackerMIL_create()

    # Dictionary to store object trackers
    trackers = {}

    # Counter for unique car IDs
    car_id_counter = 1

    # Open video file
    video_capture = cv2.VideoCapture(video_path)

    # Find the class ID for the "car" label dynamically
    car_class_label = 'car'
    car_class_id = find_class_id(classes_file, car_class_label)

    # Frame resizing parameters
    target_frame_size = (800, 600)  # Adjust the dimensions as needed

    # Frame skipping parameters
    frame_counter = 0
    skip_frames = 5

    while True:
        ret, frame = video_capture.read()
        frame_counter += 1

        if frame_counter % skip_frames != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, target_frame_size)

        # Detect objects in frame using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # List to store active object IDs in the current frame
        active_object_ids = []

        # Loop through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == car_class_id and confidence > 0.2:
                    # Draw detection box
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Check for negative dimensions
                    if endX > startX and endY > startY:
                        # Draw bounding box
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                        # Check if a tracker is already associated with this object ID
                        matched_object_id = None
                        for object_id, trk in trackers.items():
                            if trk[0].update(frame) and trk[1][0] <= startX <= trk[1][2] and trk[1][1] <= startY <= trk[1][3]:
                                matched_object_id = object_id
                                break

                        # Initialize a new tracker if no match is found
                        if matched_object_id is None:
                            # Create a new tracker for the detected car
                            tracker = cv2.TrackerMIL_create()
                            tracker.init(frame, (startX, startY, endX - startX, endY - startY))
                            trackers[car_id_counter] = (tracker, (startX, startY, endX, endY))
                            car_id_counter += 1

                        active_object_ids.append(matched_object_id)

        # Display car count
        unique_car_count = len(set(active_object_ids))
        cv2.putText(frame, "Car count: " + str(unique_car_count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display output frame
        cv2.imshow('Car counting', frame)

        # Print car count to the terminal
        print("Car count:", unique_car_count)

        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r'D:\1. APPS\Works\Akash1\Projects\car\InShot_20231218_190515911.mp4'  # Adjust the path to your video file
    count_cars(video_path)
