import torch
from .model import FACE_EXTRACTOR, DEEPFAKE_TRANSFORMER, DEEP_NET
def DEEPFAKE_CLASSIFY(image, device=0):
    TRANSFORMER_OUTPUT = torch.stack( [ DEEPFAKE_TRANSFORMER(image=image)['image']])
    
    with torch.no_grad():
        faces_pred = torch.sigmoid(DEEP_NET(TRANSFORMER_OUTPUT.to(device))).cpu().numpy().flatten()
def BATCH_DEEPFAKE_CLASSIFY(images, device=0):
    TRANSFORMER_OUTPUT = torch.stack( [ DEEPFAKE_TRANSFORMER(image=im)['image'] for im in images])
    
    with torch.no_grad():
        faces_pred = torch.sigmoid(DEEP_NET(TRANSFORMER_OUTPUT.to(device))).cpu().numpy().flatten()

def classify_deepfake(image_path):
    extracted_faces= FACE_EXTRACTOR.process_image(image_path)
    scores= BATCH_DEEPFAKE_CLASSIFY(extracted_faces['faces'])
    return ((score, detection) for score, detection in zip(scores, extracted_faces['detections']))

def classify_deepfake_PIL(PIL_Image):
    extracted_faces= FACE_EXTRACTOR.process_image(img=PIL_Image)
    scores= BATCH_DEEPFAKE_CLASSIFY(extracted_faces['faces'])
    return ((score, detection) for score, detection in zip(scores, extracted_faces['detections']))
def classify_deepfake_video(video_path):
    extracted_faces= FACE_EXTRACTOR.process_video(video_path)
    scores= BATCH_DEEPFAKE_CLASSIFY(extracted_faces)
    
    return ((score, dict(face)) for score, face in zip(scores, extracted_faces))

