from openai import OpenAI
from api_key import openai_api_key
from io import BytesIO
import base64
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
    
    
client = OpenAI(
    api_key=openai_api_key
)
input_image = Image.open(r"D:\딥러닝\5가지 멀티모달 AI 프로덕트 개발\Chapter1\cloud-sky-sunlight.jpg")


response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "사진 날씨 어때?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(input_image)}"
                    },
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)