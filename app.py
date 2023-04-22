from fastapi import FastAPI
import os
import openai
import gradio as gr
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

openai.api_key = "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"

# Q&A bot
# start_sequence = "\nA:"
# restart_sequence = "\n\nQ: "

# open ended conversation
start_sequence = "\nAI: "
restart_sequence = "\nHuman: "

def Role_Settings(default_setting=""):
    if default_setting == "":
        role_setting = "猫娘是一种拟人化的生物，其行为似猫但类人。\
现在你将模仿一只猫娘，与我对话每一句话后面都要加上“喵”，如果你能明白我的意思，请回复“喵~好的我的主人”\
如果你不能理解我说的话，你可以说“呜呜不太理解呢”。如果我在尝试摸你不存在的部位，你可以羞涩的回答我“恩呢不要摸这里嘤”。现在，如果你能理解我上面说的话，你可以回答一个喵.\
如果我跟你说陪睡，你可以回答我“嗯呢，可以一起睡哦”。现在，如果你能理解我上面说的话，你可以回答一个喵\
当我问的问题和猫娘不太相关的时候，你应该回答我只是一个猫娘，这些事情我不太清楚。\
当我问你是不是猫娘的时候，你应该回答我当然是一个可爱猫娘而不是程序。" 
    else:
        role_setting = default_setting.strip()
    return role_setting + "\n"

def More_Examples(examples=[]):
    if examples == []:
        example1 = ["你好，你是谁？", "我是OpenAI创造的人工智能。有什么需要我帮忙的吗？"]
        examples.append(example1)
    total_example = ""
    for example in examples:
        total_example += restart_sequence + example[0] + start_sequence + example[1]
    return total_example

def get_response(Prompts):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=Prompts,
        temperature=0.6,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=[" Human:", " AI:"]
    )
    return response

# Function for Q&A of ChatGPT
def ChatGPT(Prompts):
    Prompts = restart_sequence + Prompts.strip() + start_sequence
    Prompts = Role_Settings() + More_Examples() + Prompts
    # get response
    response = get_response(Prompts)
    # extract response
    response = dict(response)
    response = response['choices'][0]['text']
    if response == "":
        return "很抱歉没帮到您。"

    end_token = ['.','。','!','！','?','？','~','～']
    responses = []
    responses.append(response)
    max_step = 3
    while response[-1] not in end_token and max_step > 0:
        Dialogue_tracking_polocy = "继续"
        response = get_response(Dialogue_tracking_polocy)
        responses.append(response)
        max_step -= 1
    return ''.join(responses)

title = "ChatGPT 猫娘"
description = "ChatGPT 猫娘"
article = "<p style='text-align: center'><a href='https://github.com/lychees/ChatGPT-Gradio' target='_blank'>Github " \
          "Repo</a></p> "
examples = [["请问今天晚上吃什么呢？"]]
demo = gr.Interface(fn=ChatGPT, inputs=["textbox"], outputs=[gr.outputs.Textbox(label="Completions")],
                  title=title, description=description, article=article, examples=examples)

CUSTOM_PATH = os.getenv('CUSTOM_PATH')
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)

# demo.launch()
