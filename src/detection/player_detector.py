import cv2 
import numpy as np
from ultralytics import YOLO
import os


class PlayerDetector:
    def __init__(self,  use_custom_model = False, model_path = None): #load the YOLO model
        if use_custom_model and model_path and os.path.exists(model_path): #check if we can load custom model
            self.model =YOLO(model_path) #load roboflow model
            self.is_custom = True
            print("Custom model loaded")
        else:
            self.model = YOLO('yolo11s.pt')#load Yolo model
            self.is_custom = False
            print("YOLO model loaded")
    
        
    def detect_players(self, frame): #detect players in the frame
        if self.is_custom:
            result = self.model(frame, conf = 0.35)
        else:
            result = self.model(frame, classes = [0], conf = 0.3)
        print(f"Detected {len(result[0].boxes)} players")
        return result[0].boxes

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

            boxes = self.detect_players(frame)
            frame = self.draw_boxes(frame, boxes)
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()
        print("Processing complete. Output saved to", output_path)

 
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
        for box in boxes: #iterate through the boxes and draw them
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
    
if __name__ == "__main__":
    detector = PlayerDetector()
    video_path = '/Users/noe/soccer-coach-ai/data/videos/video.mp4'
    custom_model_path = '/Users/noe/soccer-coach-ai/models/best.pt'
    if os.path.exists(custom_model_path):
        detector = PlayerDetector(use_custom_model=True, model_path=custom_model_path)
        output_path = '/Users/noe/soccer-coach-ai/data/videos/output_video.mp4'
        detector.process_video(video_path, output_path)