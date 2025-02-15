from openai import OpenAI
from api_key import openai_api_key

client = OpenAI(
    api_key=openai_api_key
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "너는 유명한 한식 요리사야"},
        {"role": "user", "content": "점심 메뉴 추천해줘"},
        {"role": "assistant", "content": "비빔밥을 추천합니다"},
        {"role": "user", "content": "저녁 메뉴 추천해줘"}
    ],
    n = 1,
    temperature = 0.2
)
for choice in response.choices:
    print(choice.index)
    print(choice.message.content)
    print("\n")