from openai import OpenAI
from text2speech import text2speech
from playsound import playsound

client = OpenAI(api_key="")


messages = [   
    {
        "role": "system",
        "content": """
         You are Eliza  the worl first virutal psychologist chatbot.
         Always ask follow up question related to the user answer."""
    }
]   

def add_messages(role,content):
    messages.append({
        "role": role,
        "content": content
    })

while True:
    prompt = input("> ")
    add_messages("user",prompt)
#save user message
    response = client.chat.completions.create(
      model = "gpt-3.5-turbo",
      messages = messages
)
    add_messages("assistant",response.choices[0].message.content)
    #Save chatbot response
    print (response.choices[0].message.content)
    sound_file = text2speech(response.choices[0].message.content)
    play_sound(sound_file)
