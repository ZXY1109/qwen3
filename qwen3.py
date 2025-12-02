import torch
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和分词器
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    ".",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    ".",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16  
)
print("模型加载完成！")

# 2. 文本拆分函数
def split_text(text, chunk_size=2000, chunk_overlap=100):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if start > 0:
            start -= chunk_overlap
        chunks.append(text[start:end])
        start = end
    return chunks

# 3. 读取PDF并拆分
def load_long_document(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return split_text(text)  # 直接返回拆分后的文本块

# 4. 文档分析函数
def analyze_contract(doc_chunks):
    # 只分析1个核心点，且文本再缩短到1000字内
    prompt = f"""请分析以下文档，总结人工智能核心特征：
文档文本：{doc_chunks[0][:1000]}  # 仅取前1000字
回答："""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # 生成更短的内容
        do_sample=False,  # 关闭采样，用最快的贪心解码
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. 运行主流程
if __name__ == "__main__":
    # 替换为你的PDF绝对路径
    pdf_path = "C:/Users/颖/qwen3-models/Qwen3-8B/Qwen3-8B/AI.pdf"
    doc_chunks = load_long_document(pdf_path)
    print(f"文档处理完成，共拆分为{len(doc_chunks)}段！")
    
    # 执行分析
    result = analyze_contract(doc_chunks)
    print("\n分析结果：")
    print(result)