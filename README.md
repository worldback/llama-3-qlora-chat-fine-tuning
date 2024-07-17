# llama-3-qlora-chat-fine-tuning

基于meta-llama/Meta-Llama-3-8B 使用 qlora 进行会话分布式微调

## 数据来源和处理

数据来源是hugging face上的数据集：stingning/ultrachat
经过过滤后，得到的数据。具体处理过程参考 preprocess_data.ipynb。

处理的数据百度云链接:    [data](https://pan.baidu.com/s/15fESwenH6N9ICUTtKlMoFQ?pwd=744w)   

## 训练

先安装环境:
pip install -r requirementx.txt

在4块A800-80G上运行:

```
accelerate launch --config_file deepspeed_config_z3_qlora.yaml  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Meta-Llama-3-8B" \
--dataset_name "1024_train_and_test" \
--max_seq_len 1024 \
--num_train_epochs 3 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "steps" \
--eval_steps 300 \
--save_strategy "steps" \
--save_steps 300 \
--bf16 True \
--learning_rate 4e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "trainer-llama-3-chat" \
--per_device_train_batch_size 30 \
--per_device_eval_batch_size 30 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 64 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--load_best_model_at_end True \
--bnb_4bit_quant_storage_dtype "bfloat16" \
--load_best_model_at_end True \
--metric_for_best_model "loss" \
--save_total_limit 5
```

根据GPU情况适当调整参数。但请注意:   

max_seq_len: 若需要修改最大序列长度，参考preprocess_data.ipynb重新生成数据。已处理好的数据token化后基本不超过1024。若max_seq_len变大如2048，请重新处理数据，挑选合适的长度。   
model_name_or_path： 由于utils.py中只针对llama-3设定了chat_template, 因此除非添加对应的聊天模板，否则底模不要更改。如果底模是仅完成模型（即没有经过指令微调、聊天微调如llama-2-instruction等）则没关系。

在deepspeed_config_z3_qlora.yaml 修改num_processes为GPU数量。   

大概需要13h。 

## 测试

```
python test.py --path your-fine-tuning-model-path
```

![image](https://github.com/user-attachments/assets/c054d458-2a90-4a13-b027-3cec2da08524)



## 参考

https://medium.com/@xuebinbin12/fine-tuning-chat-based-llm-with-multi-turn-conversational-data-part-i-d8c64d01a20d   
https://github.com/pacman100/LLM-Workshop.git
