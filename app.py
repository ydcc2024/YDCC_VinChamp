from model.deepfake_detector import classify_deepfake_PIL, classify_deepfake_video

import requests
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import cv2
from numpy.linalg import norm

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        height, width= img.shape[0], img.shape[1]
        img_grey = cv2.cvtColor(img[height//4:height*3//4, width//4:width*3//4,:], cv2.COLOR_BGR2GRAY)
        return img_grey.std()/127
    else:
        # Grayscale
        return np.average(img)/127
  
def get_text_color(img):
  bn= brightness(img)
  if bn > 0.5:
    return 'black'
  return 'white'
def draw_text_and_rectangle(image, text, xyxy, text_color="black", outline_color="red", fill_color=None, in_place=False):
  """
  Opens an image, draws a rectangle and text above it, and returns the modified image.

  Args:
      image_path: Path to the image file.
      text: The text to be drawn above the rectangle.
      xywh: Tuple containing [x, y, width, height] of the rectangle in xywh format.
      text_color: Color for the text (default: black).
      outline_color: Color for the rectangle's outline (default: red).
      fill_color: Color to fill the rectangle (default: None for outline only).

  Returns:
      The modified PIL Image object with the drawn rectangle and text.
  """
  if not in_place:
    image= image.copy()
  draw = ImageDraw.Draw(image)
  draw.rectangle(xyxy, outline=outline_color, fill=fill_color, width=2)
  font = ImageFont.truetype('./arial.ttf', 28, encoding="unic")
  x0,y0,x1,y1= xyxy
  draw.text((x0*0.98+x1*0.02, y0*.985 +y1*.015), text, font= font, fill=text_color)

  return image

def predict_image(image):
  results= classify_deepfake_PIL(image)
  for score, detection in results:
    ymin,xmin,ymax,xmax= detection[:4]
    image= draw_text_and_rectangle(image, 
                            get_text(score),
                            (xmin,ymin,xmax,ymax), in_place=True)
    
  return image
def get_text(score):
  return f"Real {score*100:.3f}% conf" if score >= 0.5 else f"deepfake {100-score*100:.3f}% conf"


def predict_image(image):
  results= classify_deepfake_PIL(image)
  for score, detection in results:
    ymin,xmin,ymax,xmax= detection[:4]
    image=draw_text_and_rectangle(image, 
                            get_text(score),
                            (xmin,ymin,xmax,ymax), in_place=True)
  
  return image
def PIL_2_CV2(image):
  return np.array(image)[:,:,::-1]

def write_video(save_name,frames, fps=25):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  #print(f'shape {frames[0].shape}')
  video = cv2.VideoWriter(save_name, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))  
  # Appending the images to the video one by one 
  for frame in frames:  
      video.write(frame)
  video.release()
  return save_name

def predict_video(video_path):
  results= classify_deepfake_video(video_path)
  frames=[]
  text_color= get_text_color(PIL_2_CV2(results[0][0]))
  for frame, result in results:
    for score, detection in result:
      ymin,xmin,ymax,xmax= detection[:4]
      frame= draw_text_and_rectangle(frame, 
                            get_text(score),
                            (xmin,ymin,xmax,ymax),
                            text_color=text_color,
                            outline_color=text_color)
    frames.append(PIL_2_CV2(frame))
  
  
  save_name= f'./outputs/{datetime.time(datetime.now())}.mp4'
  write_video(save_name, frames)
  return save_name


image_if= gr.Interface(fn=predict_image,
             inputs=gr.Image(type="pil"),
             outputs=gr.Image(type="pil"),
             examples=["./samples/real.jpg", "./samples/fake.jpg", "./samples/obama.jpg","./samples/obama_real.webp"])

video_if= gr.Interface(fn= predict_video,
                       inputs=gr.Video(),
                       outputs=gr.Video(),
                       examples=['./samples/morgan_sample.mp4','./samples/vinuni_sample.mp4'])


gr.TabbedInterface(
    [image_if, video_if], ["Image", "Video"]
).launch()






