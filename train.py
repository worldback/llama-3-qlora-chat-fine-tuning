from dataclasses import dataclass, field
import os
import sys
from typing import Optional
import torch
from transformers import set_seed, Trainer, HfArgumentParser, TrainingArguments
from trl import  DataCollatorForCompletionOnlyLM
from utils import  create_and_prepare_model, create_custom_datasets, loftq_init
torch.set_printoptions(profile="full")
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={ "help": "comma separated list of target modules to apply LoRA layers to"})
    use_nested_quant: Optional[bool] = field(default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"})
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16",
        metadata={"help": "Compute dtype for 4bit base models"})
    bnb_4bit_quant_storage_dtype: Optional[str] = field(default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "Quantization type fp4 or nf4"})
    use_flash_attn: Optional[bool] = field(default=False, metadata={"help": "Enables Flash attention for training."})
    use_peft_lora: Optional[bool] = field( default=False, metadata={"help": "Enables PEFT LoRA for training."})
    use_8bit_qunatization: Optional[bool] = field(default=False, metadata={"help": "Enables loading model in 8bit."})
    use_4bit_quantization: Optional[bool] = field(default=False, metadata={"help": "Enables loading model in 4bit."})
    use_reentrant: Optional[bool] = field(default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"})
    use_unsloth: Optional[bool] = field(default=False,
        metadata={"help": "Enables UnSloth for training."})
    use_loftq: Optional[bool] = field(default=False,
        metadata={"help": "Enables LoftQ init for the LoRA adapters when using QLoRA."})
    use_loftq_callback: Optional[bool] = field(default=False,metadata={"help":
      "Enables LoftQ callback comparing logits of base model to the ones from LoftQ init. Provides better init."})
    moe_layer_name: Optional[str] = field(default=None, metadata={"help": "MOE layer name"})


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(metadata={"help": "The preference dataset to use."})
    max_seq_length: Optional[int] = field(default=1024)



def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = (
        training_args.gradient_checkpointing and not model_args.use_unsloth
    )
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    # datasets
    train_dataset, eval_dataset = create_custom_datasets(tokenizer,data_args)

    response_template = '<|start_header_id|>assistant<|end_header_id|>'
    instruction_template = '<|start_header_id|>user<|end_header_id|>'
    data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, mlm=False,
                                               response_template=response_template, tokenizer=tokenizer)
    trainer = Trainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args
    )

    trainer.accelerator.print(f"{trainer.model}")
    if model_args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    # LoftQ initialization when using QLoRA
    if model_args.use_4bit_quantization and model_args.use_loftq:
        loftq_init(
            trainer.model,
            tokenizer,
            train_dataset,
            data_args.max_seq_length,
            model_args,
        )

    # train_dataloader = trainer.get_train_dataloader()
    # idx = 0
    # for batch in train_dataloader:
    #     decode_txt = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
    #     print(decode_txt)
    #     print(batch.keys())  # 查看键，通常包括'input_ids', 'attention_mask', 'labels'等
    #     print(batch['input_ids'][0])
    #     print("------------------------------------")
    #     print(batch['labels'][0])
    #     print("——————————————————————————————————————————————")
    #     print(batch['input_ids'][0].shape, batch['labels'][0].shape)
    #     label = batch['labels'][0]  #  查看是否正确设置标签
    #     mask = label != -100
    #     filtered_tensor = label[mask]
    #     decode_txt = tokenizer.decode(filtered_tensor, skip_special_tokens=True)
    #     print('=========================================')
    #     print(decode_txt)
    #     idx += 1
    #     if idx > 1:
    #         break

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
