import gc
from dataclasses import dataclass
import torch
from datasets import  load_from_disk
from peft import LoraConfig, replace_lora_weights_loftq, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer, BitsAndBytesConfig
from trl import setup_chat_format

current_mse = float("inf")

@dataclass
class ChatMlSpecialTokens:
    bos_token: str = "<|begin_of_text|>"
    rbos_token: str = "<|start_header_id|>"
    reos_token: str = "<|end_header_id|>"
    eot_token: str = "<|eot_id|>"
    eos_token: str = "<|end_of_text|>"
    pad_token: str = "<pad>"

    @property
    def all_specail_tokens(self):
        return [self.bos_token, self.rbos_token, self.reos_token,
            self.eot_token, self.eos_token, self.pad_token]

    @property
    def system(self):
        return "system"

    @property
    def user(self):
        return "user"

    @property
    def assistant(self):
        return "assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
                "{% if loop.first %}"  # 判断是否为循环的第一个元素  
                    f"{{{{ '{self.bos_token}' + '{self.rbos_token}' + message['role'] +  '{self.reos_token}' + message['content'] + '{self.eot_token}' }}}}"
                "{% else %}"
                    f"{{{{'{self.rbos_token}' + message['role'] + '{self.reos_token}' + message['content'] + '{self.eot_token}'  }}}}"
                "{% endif %}"
                "{% if loop.last and message['role'] == 'assistant' %}"  # 判断是否为循环的最后一个元素且需要添加生成提示  
                    f"{{{{ '{self.eos_token}' }}}}"
                "{% endif %}"
            "{% endfor %}"
            "{%  if add_generation_prompt %}"
                f"{{{{ '{self.rbos_token}' + '{self.assistant}' + '{self.reos_token}' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}

def setup_chat_format(
    model,
    tokenizer,
    format= "chatml",
    resize_to_multiple_of = None,
):
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")
    chat_format = FORMAT_MAPPING[format]()
    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": chat_format.all_specail_tokens})
    tokenizer.chat_template = chat_format.chat_template
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def create_custom_datasets(tokenizer, data_args):
    raw_datasets =load_from_disk(data_args.dataset_name)
    token_train_data = SupervisedDataset(raw_datasets['train'], tokenizer)
    token_val_data = SupervisedDataset(raw_datasets['test'], tokenizer)
    return token_train_data, token_val_data


class SupervisedDataset(Dataset):

    def __init__(self, data, tokenizer):
        super(SupervisedDataset, self).__init__()
        self.data = data
        self.tokenizer= tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.tokenizer.apply_chat_template(self.data[i]['messages'], tokenize=False)
        tokenize_data = self.tokenizer(data, return_length=True, add_special_tokens=False,
                                       padding='max_length', max_length=1024, truncation=True)
        # return dict(input_ids=tokenize_data["input_ids"],
        #             attention_mask=tokenize_data["attention_mask"],
        #             length=tokenize_data["length"])
        return dict(input_ids=tokenize_data["input_ids"],
                    attention_mask=tokenize_data["attention_mask"])

def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_stype = None
    load_in_8bit = args.use_8bit_qunatization
    load_in_4bit = args.use_4bit_quantization

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_stype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_stype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    else:
        torch_dtype = quant_storage_stype if quant_storage_stype and quant_storage_stype.is_floating_point else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=load_in_8bit,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        model_max_length=data_args.max_seq_length,
        trunction=True, padding_side="right", use_fast=True
    )

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer

def get_mae(x, y):
    return (x - y).abs().mean()


def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def error_report(x, y):
    mae = get_mae(x, y)
    mse = get_mse(x, y)
    print(
        f"Mean absolute error: {mae:>8.5f}\n"
        f"Mean squared error:  {mse:>8.5f}"
    )


def loftq_init(model, tokenizer, train_dataset, max_seq_length, args):
    if args.use_loftq_callback:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=compute_dtype)
        base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        random_input_ids = torch.randint(0, len(train_dataset), size=(1,)).numpy().tolist()
        random_inputs = [train_dataset[i]['content'] for i in random_input_ids]
        random_inputs = tokenizer(random_inputs, return_tensors="pt", padding=True, truncation="max_length", max_length=max_seq_length)
        logits_base = base_model(**random_inputs).logits
        del base_model
        gc.collect()
        
        def loftq_callback(model, module_name):
            """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
            global current_mse
            logits = model(**random_inputs).logits
            mse = get_mse(logits_base, logits)
            if mse < current_mse:
                current_mse = mse
                print(f"MSE improved for module {module_name}")
                return True
            print(f"MSE did not improve for module {module_name}")
            return False
        
        replace_lora_weights_loftq(model, callback=loftq_callback)
        logits_loftq_callback = model(**random_inputs).logits
        error_report(logits_base, logits_loftq_callback)
    else:
        replace_lora_weights_loftq(model)
    

def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
