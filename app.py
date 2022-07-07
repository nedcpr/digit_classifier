import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

categories = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def classify_image(img):
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(100,100))
label = gr.outputs.Label()
title = "Sign Language Digit Classifier"
description = "A sign language digit classifier trained on a Kaggle dataset with fastai. Created as a demo for fast.ai Part 1 v5 (2022)."
examples="IMG_4590.JPG", "IMG_4883.JPG", "IMG_5495.JPG"
interpretation='default'
enable_queue=True

gr.Interface(fn=classify_image,inputs=gr.inputs.Image(shape=(100, 100)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,examples=examples).launch()
