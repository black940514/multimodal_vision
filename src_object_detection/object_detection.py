from openai import OpenAI
import sys
import os

# 상위 디렉터리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api_key import openai_api_key
from io import BytesIO
import base64
import gradio as gr
from PIL import Image


def encode_image(image):
    buffered = BytesIO()
    w, h = image.size
    if w > 512 or h > 512:
        scale = 512 / max(w, h)
        
    else :
        scale = 1.0
    resize_im = image.resize((int(w * scale), int(h * scale)))
    resize_im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str    

def obj_detection(image):
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "Given the image, answer the following question."
            },
            {
                "role": "user",
                "content":[
                    {
                        "type": "text",
                        "text":  "Question: Detect and locate the following objects in the image. \
                         Give a confidence score for each detection between 0 and 1. \
                         Give me a specific x coordinate bounding box of each object between 0 and 1. \
                         leftmost pixel is 0 and rightmost pixel is 1. \
                         Give me a specific y coordinate bounding box of each object between 0 and 1. \
                         uppermost pixel is 0 and bottommost pixel is 1. \
                    Objects: [Dog, Cat]."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    demo = gr.Interface(
        obj_detection,
        [gr.Image(type='pil')],
        "textbox",
        theme=gr.themes.Default(text_size="lg")
    )
    demo.launch()