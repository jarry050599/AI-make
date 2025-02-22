import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import re
import os
from datasets import load_dataset, Dataset
from fpdf import FPDF
from docx import Document

# 檢查 CUDA 是否可用
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 安裝必要的套件
try:
    from PyPDF2 import PdfReader
except ImportError:
    os.system("pip install PyPDF2")
    from PyPDF2 import PdfReader

try:
    from Crypto.Cipher import AES
except ImportError:
    os.system("pip install pycryptodome")
    from Crypto.Cipher import AES

# 模型與 Tokenizer 設定
MODEL_NAME = "microsoft/DialoGPT-medium"

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)  # 移除HTML標籤
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；「」『』]", "", text)  # 保留中文、英文、數字
    text = re.sub(r"\s+", " ", text)  # 移除多餘空白
    return text.strip()

# 讀取並清理資料
def load_training_data():
    cleaned_data = []
    if not os.path.exists("train_data"):
        response_text.insert(tk.END, "未找到 'train_data' 資料夾，請將資料放置在該資料夾內。\n")
        return cleaned_data

    for filename in os.listdir("train_data"):
        file_path = os.path.join("train_data", filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                cleaned_data.extend([clean_text(line) for line in lines if len(clean_text(line)) > 10])
        elif filename.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        cleaned_data.extend([clean_text(line) for line in text.split("\n") if len(clean_text(line)) > 10])
            except Exception as e:
                response_text.insert(tk.END, f"無法處理 PDF 檔案 {filename}：{str(e)}\n")
        elif filename.endswith(".docx"):
            try:
                doc = Document(file_path)
                for para in doc.paragraphs:
                    cleaned_data.append(clean_text(para.text))
            except Exception as e:
                response_text.insert(tk.END, f"無法處理 DOCX 檔案 {filename}：{str(e)}\n")

    return cleaned_data

# 訓練模型
def train_model():
    try:
        cleaned_data = load_training_data()
        if not cleaned_data:
            response_text.insert(tk.END, "未找到有效的訓練資料。\n")
            return

        dataset = Dataset.from_dict({"text": cleaned_data})

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token  # 設置 padding token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # 切割訓練和驗證集
        split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        training_args = TrainingArguments(
            output_dir="./output",
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

        response_text.insert(tk.END, "開始訓練模型...\n")
        trainer.train()
        model.save_pretrained("./trained_model")
        tokenizer.save_pretrained("./trained_model")
        response_text.insert(tk.END, "模型訓練完成，已保存至 './trained_model'。\n")
    except Exception as e:
        response_text.insert(tk.END, f"模型訓練失敗：{str(e)}\n")

# 載入模型
def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./trained_model" if os.path.exists("./trained_model") else MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("./trained_model" if os.path.exists("./trained_model") else MODEL_NAME).to(DEVICE)

# 生成回應
def generate_response():
    user_input = user_input_text.get("1.0", tk.END).strip()
    if not user_input:
        return

    inputs = tokenizer(user_input, return_tensors="pt", padding=True).to(DEVICE)
    output = model.generate(
        **inputs, max_new_tokens=200, do_sample=True, temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_text.insert(tk.END, f"你: {user_input}\nLLM: {response}\n{'-'*40}\n")
    user_input_text.delete("1.0", tk.END)

# GUI 介面
root = tk.Tk()
root.title("簡單 LLM GUI")
root.geometry("600x600")

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

# 訓練模型按鈕
def train_data():
    train_model()

train_button = tk.Button(root, text="訓練模型", command=train_data)
train_button.pack(pady=5)

# 啟動應用
def start_app():
    response_text.insert(tk.END, "正在載入模型，請稍候...\n")
    root.update()
    try:
        load_model()
        response_text.insert(tk.END, "模型載入完成！開始對話吧！\n\n")
    except Exception as e:
        response_text.insert(tk.END, f"模型載入失敗：{str(e)}\n\n")

# 啟動應用
start_app()
root.mainloop()
