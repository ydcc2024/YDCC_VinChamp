# VinChamp - April 14, 2024 (YDCC 2024)

import torch
from .model import FACE_EXTRACTOR, DEEPFAKE_TRANSFORMER, DEEP_NET
import cv2
from PIL import Image
def DEEPFAKE_CLASSIFY(image, device='cpu'):
    TRANSFORMER_OUTPUT = torch.stack( [ DEEPFAKE_TRANSFORMER(image=image)['image']])
    
    with torch.no_grad():
        faces_pred = torch.sigmoid(DEEP_NET(TRANSFORMER_OUTPUT.to(device))).cpu().numpy().flatten()
    return faces_pred
def BATCH_DEEPFAKE_CLASSIFY(images, device='cpu'):
    TRANSFORMER_OUTPUT = torch.stack( [ DEEPFAKE_TRANSFORMER(image=im)['image'] for im in images])
    
    with torch.no_grad():
        faces_pred = torch.sigmoid(DEEP_NET(TRANSFORMER_OUTPUT.to(device))).cpu().numpy().flatten()

    return faces_pred
def classify_deepfake(image_path):
    extracted_faces= FACE_EXTRACTOR.process_image(image_path)
    scores= BATCH_DEEPFAKE_CLASSIFY(extracted_faces['faces'])
    return ((score, detection) for score, detection in zip(scores, extracted_faces['detections']))

def classify_deepfake_PIL(PIL_Image):
    extracted_faces= FACE_EXTRACTOR.process_image(img=PIL_Image)
    scores= BATCH_DEEPFAKE_CLASSIFY(extracted_faces['faces'])
    #print(extracted_faces)
    return ((score, detection) for score, detection in zip(scores, extracted_faces['detections']))


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    success, img = cap.read()
    fno = 0
    frames=[]
    while success:
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        success, img = cap.read()
    cap.release()
    return frames

def classify_deepfake_video(video_path):
    extracted_frames= extract_frames(video_path)
    processed_frames=[]
    for frame in extracted_frames:
        try:
            results= list(classify_deepfake_PIL(frame))
        except Exception as e:
            print(e)
            score=0
            detection=(0,0,0,0)
            pass
        finally:
            #print(score, detection)
            
            processed_frames.append((frame, results))
    
    return processed_frames

