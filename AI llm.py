import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型與 Tokenizer 設定
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

def load_model():
    global model, tokenizer
    import os
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        raise EnvironmentError("Please set the HUGGINGFACE_TOKEN environment variable with your Hugging Face token.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", load_in_4bit=True, token=token
    )

def generate_response():
    user_input = user_input_text.get("1.0", tk.END).strip()
    if not user_input:
        return

    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs, max_new_tokens=200, do_sample=True, temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_text.insert(tk.END, f"你: {user_input}\nLLM: {response}\n{'-'*40}\n")
    user_input_text.delete("1.0", tk.END)

# GUI 介面
root = tk.Tk()
root.title("簡單 LLM GUI")
root.geometry("600x500")

# 歡迎標題
welcome_label = tk.Label(root, text="簡單 LLM 聊天機器人", font=("Arial", 16))
welcome_label.pack(pady=10)

# 對話輸出框
response_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15)
response_text.pack(padx=10, pady=10)
response_text.insert(tk.END, "歡迎！輸入文字開始對話。\n\n")

# 使用者輸入框
user_input_text = tk.Text(root, height=3, width=70)
user_input_text.pack(padx=10, pady=5)

# 生成回應按鈕
generate_button = tk.Button(root, text="生成回應", command=generate_response)
generate_button.pack(pady=5)

# 載入模型
def start_app():
    response_text.insert(tk.END, "正在載入模型，請稍候...\n")
    root.update()
    load_model()
    response_text.insert(tk.END, "模型載入完成！開始對話吧！\n\n")

# 啟動應用
start_app()
root.mainloop()
# 新增功能：清除對話框內容
def clear_conversation():
    response_text.delete("1.0", tk.END)

# 新增功能：保存對話到檔案
def save_conversation():
    with open("conversation_log.txt", "w", encoding="utf-8") as f:
        f.write(response_text.get("1.0", tk.END))
    response_text.insert(tk.END, "\n對話已保存到 conversation_log.txt\n")

# 新增按鈕框架
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

clear_button = tk.Button(button_frame, text="清除對話", command=clear_conversation)
clear_button.pack(side=tk.LEFT, padx=5)

save_button = tk.Button(button_frame, text="保存對話", command=save_conversation)
save_button.pack(side=tk.LEFT, padx=5)