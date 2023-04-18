from transformers import AutoTokenizer, AutoModel
import torch
from peft import PeftModel
import json
from cover_alpaca2jsonl import format_example


# model_train = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map='auto')
# 12G以上的显存 FP16
model_train = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# 8G显存 INT8量化
# model_train = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()
model_train = PeftModel.from_pretrained(model_train, "./output/")

# CPU模式  16G内存（太慢了）
# model_original = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
# 8G显存 INT8量化s
model_original = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda()


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
instructions = json.load(open("data/alpaca_data.json"))

answers = []

with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        
        # 生成微调的数据
        input_ids_gpu = torch.LongTensor([ids]).to('cuda')
        out = model_train.generate(
            input_ids=input_ids_gpu,
            max_length=1500,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)

        # 生成未微调的答案
        input_ids_cpu = torch.LongTensor([ids])
        out_original = model_original.generate(
            input_ids=input_ids_gpu,
            max_length=1500,
            do_sample=False,
            temperature=0
        )
        out_text_original = tokenizer.decode(out_original[0])
        answer_original = out_text_original.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer_original'] = answer_original
        print('\n未训练答案数据：',answer_original,'\n')

        print(f"训练的答案数据： {idx+1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})