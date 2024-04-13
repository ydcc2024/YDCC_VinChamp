from model.deepfake_detector import classify_deepfake_PIL, classify_deepfake_video

import requests
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
def draw_text_and_rectangle(image, text, xyxy, text_color="black", outline_color="red", fill_color=None):
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
  draw = ImageDraw.Draw(image)
  draw.rectangle(xyxy, outline=outline_color, fill=fill_color)

  font = ImageFont.truetype("arial.ttf", 20)  # Replace with your desired font path and size
  text_width, text_height = draw.textsize(text, font=font)
  x0,y0,x1,y1= xyxy
  text_x = int((x0 + x1 - text_width) / 2)  
  text_y = y0 - text_height  
  
  draw.text((text_x, text_y), text, font=font, fill=text_color)

  return image

def predict_image(image):
  results= classify_deepfake_PIL(image)
  for score, detection in results:
    ymin,xmin,ymax,xmax= detection[:4]
    draw_text_and_rectangle(image, 
                            f'{f"Real {score}% confidence" if score >= 0.5 else f"Fake {1-score}% confidence"}',
                            (xmin,ymin,xmax,ymax))
    
  return image
  
gr.Interface(fn=predict_image,
             inputs=gr.Image(type="pil"),
             outputs=gr.Image(type="pil"),
             examples=["./samples/real.jpg", "./samples/fake.jpg"]).launch()







