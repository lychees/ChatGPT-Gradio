import os
os.system('pip install openai')
# os.system('pip install transformers')
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
"""
# generate prompt
tokenizer = AutoTokenizer.from_pretrained("merve/chatgpt-prompts-bart-long")
model = AutoModelForSeq2SeqLM.from_pretrained("merve/chatgpt-prompts-bart-long", from_tf=True)

def generate(prompt):
    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

examples = [["photographer"], ["developer"]]
ans = generate(examples[0][0])
"""

def Role_Settings(default_setting=""):
    if default_setting == "":
        role_setting = "我想让你扮演一名医生，为疾病想出创造性的治疗方法。" \
                       "你应该能够推荐传统药物、草药和其他天然替代品。在提供建议时，你还需要考虑患者的年龄、生活方式和病史。"
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

title = "Q&A demo of ChatGPT."
description = "Q&A demo of ChatGPT."
article = "<p style='text-align: center'><a href='https://github.com/loveleaves' target='_blank'>ChatGPT Github " \
          "Repo</a></p> "
examples = [["为患有关节炎的老年患者提出以整体治疗方法为重点的治疗方案。"]]
io = gr.Interface(fn=ChatGPT, inputs=["textbox"], outputs=[gr.outputs.Textbox(label="Completions")],
                  title=title, description=description, article=article, examples=examples,
                  allow_flagging=False, allow_screenshot=False)


io.launch()