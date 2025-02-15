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

def classify_image(image):
    client = OpenAI(
        api_key=openai_api_key
    )
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": "Given the image, answer the following questions using the specified format."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Question: Classify the image. \
                            Choices: [burglary, explosion, gunshot, fighting, none of the above.] \
                            Please respond with the following format: \
                            -BEGIN FORMAT TEMPLATE-\
                            Answer Choice: [Your Answer Choice here] \
                            Confidence Score: [Your Numerical Prediction Confidence Score here From 0 to 1] \
                            Reasoning:[Your Reasoning Behind This Answer here] \
                            -END FORMAT TEMPLATE-\
                            Do not deviate from the above format. \
                            Repeat the format template for the answer. \
                            Do not include -BEGIN FORMAT TEMPLATE- or -END FORMAT TEMPLATE- in your response."
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
        classify_image,
        [gr.Image(type='pil')],
        "textbox",
        theme=gr.themes.Default(text_size="lg")
    )
    demo.launch()