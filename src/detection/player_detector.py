import cv2 
import numpy as np
from ultralytics import YOLO
import os
import supervision as sv


class PlayerDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path): #check if we can load custom model
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        
        self.model =YOLO(model_path) #load trained model
        print("Custom model loaded")
        self.tracker = sv.ByteTrack()
        
    def detect_players(self, frame): #detect players in the frame
        result = self.model(frame, conf = 0.35)
        print(f"Detected {len(result[0].boxes)} players")
        return result[0].boxes
    def detect_and_track(self, frame): #detect and track players in the frame
        result = self.model(frame, conf = 0.35) #perform detection

        detections = sv.Detections.from_ultralytics(result[0]) #convert to supervision format
        tracked_detections = self.tracker.update_with_detections(detections) #update tracker with detections

        return tracked_detections #return tracked detections
    
    def process_video(self, video_path, output_path): #process the video and save the output
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {total_frames} frames at {fps} FPS, resolution: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tracked_detections = self.detect_and_track(frame)
            frame = self.draw_tracked_objects(frame, tracked_detections)
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()
        print("Processing complete. Output saved to", output_path)

    def draw_tracked_objects(self, frame, tracked_detections): #draw tracked objects on the frame
        if len(tracked_detections) == 0:
            return frame
        
        colors = {
            0: (0, 255, 0),  # Green for ball
            1: (255, 0, 0),  # Blue for gk classes
            2: (0, 0, 255),   # Red for player classes
            3: (255, 255, 0) # Yellow for ref classes
        }

        class_names = {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee'
            }

        for detection in tracked_detections:
            #Extract data from supervision format
            bbox = detection[0]
            confidence = detection[2]
            class_id = int(detection[3])
            tracker_id = detection[4] if detection[4] is not None else -1

            x1, y1, x2, y2 = map(int, bbox) # Convert to integers

            color = colors.get(class_id, (0, 255, 0))
            label = class_names.get(class_id, 'unknown')

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if tracker_id != -1:
                text = f'{label} ID:{tracker_id} {confidence:.2f}'
            else:
                text = f'{label} {confidence:.2f}'
            
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

            

'''
    def draw_boxes(self, frame, boxes): #draw bounding boxes on the frame
        if boxes is None or len(boxes) == 0:
            return frame
        
        colors = {
            0: (0, 255, 0),  # Green for player class
            1: (255, 0, 0),  # Blue for gk classes
            2: (0, 0, 255),   # Red for ref classes 
            3: (255, 255, 0), # Yellow for ball classes
            'default': (0, 255, 0) # Green for default
        }

        class_names = {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee',
            'default': 'object'
        }
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            if hasattr(box, 'cls') and box.cls is not None:
                class_id = int(box.cls[0].cpu().numpy())
                color = colors.get(class_id, colors['default'])
                label = class_names.get(class_id, class_names['default'])
            else:
                color = colors['default']
                label = class_names['default']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f'{label} {confidence:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame
    '''
if __name__ == "__main__":
    video_path = '/Users/noe/soccer-coach-ai/data/videos/video.mp4'
    model_path = '/Users/noe/soccer-coach-ai/models/best.pt'
    if os.path.exists(model_path):
        detector = PlayerDetector(model_path=model_path)
        output_path = '/Users/noe/soccer-coach-ai/data/outputs/output_video.mp4'
        detector.process_video(video_path, output_path)