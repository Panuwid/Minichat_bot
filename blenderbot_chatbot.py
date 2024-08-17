from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

def chat_with_blenderbot(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # ให้โมเดลสร้างคำตอบจากข้อความที่ป้อนเข้า
    reply_ids = model.generate(**inputs)
    
    # แปลงคำตอบของโมเดลกลับมาเป็นข้อความที่มนุษย์อ่านได้
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    return bot_reply

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
    
    response = chat_with_blenderbot(user_input)
    
    print(f"Bot: {response}")
