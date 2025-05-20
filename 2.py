from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "正面" if prediction == 1 else "负面"
    except Exception as e:
        print(f"预测时发生错误: {e}")
        return "未知"

# 测试输出
print("\n测试输出：")
text1 = "看完只觉得浪费了两个小时，再也不想看第二遍。"
text2 = "味道非常一般，跟评论区说的完全不一样。"
print(f"影评分类结果：{predict_sentiment(text1)}")
print(f"外卖评价分类结果：{predict_sentiment(text2)}")