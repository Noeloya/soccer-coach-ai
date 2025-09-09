import cv2 
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self): #load the YOLO model
        self.model = YOLO('yolo11s.pt')
    
    def detect_players(self, frame): #detect players in the frame
        result = self.model(frame, classes = [0], conf = 0.15)
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
        print(f"pProcessing video: {total_frames} frames at {fps} FPS, resolution: {width}x{height}")

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
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return frame
    
if __name__ == "__main__":
    detector = PlayerDetector()
    video_path = '/Users/noe/soccer-coach-ai/data/videos/video.mp4'
    output_path = '/Users/noe/soccer-coach-ai/data/outputs/output.mp4'
    detector.process_video(video_path, output_path)