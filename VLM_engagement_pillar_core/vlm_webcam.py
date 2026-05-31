# requirements.txt:
# opencv-python==4.13.0.92
# torch==2.10.0
# torchcodec==0.10.0
# torchvision==0.25.0
# transformers==5.4.0
# qwen-vl-utils==0.0.14
# pandas==3.0.1

import os
import re
import json
import time
import tempfile
import argparse
import threading
import queue
from pathlib import Path
import cv2
import torch
import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
#from torchcodec.decoders import VideoDecoder

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

ENGAGEMENT_PROMPT = """
Task: Perform a deep affective analysis of the student's engagement.
Focus on identifying both 'Gains' (Laughing) and 'Losses' (Tired/Bored/Sleepy) of engagement.
If there is nobody in the video, respond with "Nobody" for state and "None" for level.

### CATEGORY DEFINITIONS:
- GAIN: Positive high-energy states (Laughing, Smiling, Enthusiastic nodding).
- STEADY: Neutral attentive focus (Steady gaze, Still head, Calm).
- LOSS: Low-energy or distracted states (Yawning, Resting head on hand, Closing eyes, Looking away).
- NONE: No person detected in the video.

### OUTPUT FORMAT:
You MUST respond strictly in the following JSON format:
{
  "state": "Choose one: [Laughing, Smiling, Attentive, Bored, Tired, Sleeping, Distracted, Normal, Nobody]",
  "level": "Choose one: [High, Mid, Low, None]",
  "primary_cue": "Brief description of the strongest visual feature (e.g., 'deep yawning', 'eyes scanning screen')",
}
"""

q = queue.Queue()
results_list = []

class VideoConsumer(threading.Thread):
    def __init__(self, queue, device_input, cache_dir):
        super().__init__(daemon=True)
        self.queue = queue
        self.device = device_input
        self.cache_dir = cache_dir

        # Load model and processor
        print(f"Loading model on {self.device}...")
        if self.device == 'cpu':
            device_map = {'': 'cpu'}
            torch_dtype = torch.float32
            device = 'cpu'
        else:
            device_map = 'auto'
            torch_dtype = 'auto'
            device = 'cuda'
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, device_map=device_map, cache_dir=self.cache_dir)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=self.cache_dir)

    def run(self):
        while True:
            # Process the video clip and return the engagement analysis
            clip_path = self.queue.get()  # Get the video path from the queue
            if clip_path is None:
                return
            print(f"Processing clip: {clip_path}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ENGAGEMENT_PROMPT},
                        {"type": "video", "video": clip_path},
                    ],
                }
            ]

            #decoder = VideoDecoder(clip_path, device=device)
            #valid_keys = ["duration_seconds", "width", "height", "num_frames","average_fps", "pixel_aspect_ratio"]
            #video_metadata_dict = {k: v for k, v in vars(decoder.metadata).items() if k in valid_keys}
            #print(f"Decoded video {clip_path} with metadata {decoder.metadata}")
            #use decoder.metadata as parameter for the processor.
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)# "cuda" or "cpu" depending on the main function argument
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=250)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            try:
                json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = {"state": "error", "level": "error"}
            except Exception as e:
                print(f"Parsing error: {e}")
                data = {"state": "error", "level": "error",}

            row = {'full_video_path': clip_path, **data}
            results_list.append(row)

            print(data)
            q.task_done()

def main():
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--interval', type=float, default=10, help='Interval in seconds for each video clips')
    parser.add_argument("--output", type=Path, default=".",help="Output path")
    parser.add_argument("--display", action="store_true", help="Show frames in a window")
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()
    
    # Create output directory and subdirectories for cache and video clips
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    cache_dir = os.path.join(args.output, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    videos_dir = os.path.join(args.output, "video_clips")
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    video_date_str = time.strftime("%Y%m%d-%H%M%S")
    video_date_dir = os.path.join(videos_dir, video_date_str)
    if not os.path.exists(video_date_dir):
        os.makedirs(video_date_dir)

    # Read prompt from file or create it with default prompt if it doesn't exist
    global ENGAGEMENT_PROMPT
    prompt_file_path = os.path.join(args.output, "engagement_prompt.txt")
    if not os.path.exists(prompt_file_path):
        with open(prompt_file_path, "w") as f:
            f.write(ENGAGEMENT_PROMPT)
    with open(prompt_file_path, "r") as f:
        ENGAGEMENT_PROMPT = f.read()

    # Initialize webcam and get properties
    cap = cv2.VideoCapture(args.camera_index)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error opening the webcam.")
        return
    
    # Define a the duration of each clip (frames per clip)
    frames_per_clip = fps * args.interval
    clip_number = 1
    frame_count = 0
    filename = None
    prev_filename = None
    writer = None

    # Start the consumer thread to process video clips
    consumer = VideoConsumer(q, args.device, cache_dir)
    consumer.start()

    print("Recording webcam... Press 'q' to stop")
    # Loop
    try:
        while True:
            # Capture frame-by-frame
            ok, frame = cap.read()
            if not ok:
                print("Error capturing frame.")
                break

            # Create new VideoWriter for each clip, with a duration of "interval" seconds
            if frame_count % frames_per_clip == 0:
                if writer is not None:
                    writer.release()
                if filename is not None:
                    prev_filename = filename
                filename = os.path.join(video_date_dir, f"clip_{clip_number:03d}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                print(f"Started recording {filename}")
                clip_number += 1
                # Process the previous clip while recording the next one
                if prev_filename is not None:
                    ### Launch process_clip in a new thread to avoid blocking the main loop
                    ### process_clip(prev_filename, model, processor, device)
                    # Put the previous clip filename in the queue for the consumer thread to process
                    q.put(prev_filename)
            writer.write(frame)
            frame_count += 1
            
            # Display the frame
            if args.display:
                cv2.imshow("Webcam", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release camera and close windows on exit
        #consumer.join()  # Wait for the consumer thread to finish processing
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        # Save results to CSV
        df = pd.DataFrame(results_list)
        output_csv_path = os.path.join(video_date_dir, f"VLM_results.csv")
        df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()
