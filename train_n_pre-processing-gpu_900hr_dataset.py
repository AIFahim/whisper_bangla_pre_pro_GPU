import os
import re
import torch
import functools
import json
import itertools
# import multiprocessing as mp
import torch.multiprocessing as mp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, DatasetDict, Dataset, Audio
from prepare_gpu import prepare_dataset
from train_val_df_gen import Train_Val_df
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Union
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperConfig,
    WhisperModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

import mlflow
import mlflow.pytorch
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])
        
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def main():
    # Configuration and setup
    os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "1234567"
    abs_path = os.path.abspath(".")
    base_dir = os.path.dirname(abs_path)
    model_name = "openai/whisper-small"
    language = "Bengali"
    task = "transcribe"  # transcribe or translate
    NUM_PROC = 1
    NUM_CHUNKS = 1
    NUM_EPOCHS = 25
    TOTAL_FILES = 0

    print(f"\n\n Loading {model_name} for {language} to {task}...this might take a while.. \n\n")
    
    # Parameters
    output_dir = "./"
    overwrite_output_dir = True
    


    with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        print(len(json_data.values()))
        TOTAL_FILES = len(json_data.values())
        json_data = dict(itertools.islice(json_data.items(), 0 * len(json_data.keys()) // NUM_CHUNKS, 1 * len(json_data.keys()) // NUM_CHUNKS))
        
        print(TOTAL_FILES)
        print(len(json_data))


    per_device_train_batch_size = 8
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    dataloader_num_workers = 2

    max_steps = math.ceil((TOTAL_FILES // NUM_CHUNKS) / (per_device_train_batch_size * gradient_accumulation_steps)) * NUM_EPOCHS  # 3000


    gradient_checkpointing = False
    evaluation_strategy = "steps"
    eval_steps = 100
    save_strategy = "steps"
    save_steps = 100
    save_total_limit = 1
    learning_rate = 1e-5
    lr_scheduler_type = "cosine"
    warmup_steps = 888
    logging_steps = 1
    weight_decay = 0
    dropout = 0.1
    load_best_model_at_end = True
    metric_for_best_model = "cer"
    greater_is_better = False
    bf16 = True
    tf32 = True
    generation_max_length = 448
    predict_with_generate = True
    push_to_hub = True
    freeze_feature_encoder = False
    early_stopping_patience = 10
    apply_spec_augment = True
    loop_train_dataset = 2
    loop_val_dataset = 1

    # Generate train and validation data
    # tran_val_df = Train_Val_df()
    # train_data, dev_data = tran_val_df.generate_df()
    #Call the generate_df_from_json() method on the Train_Val_df class directly
    train_df, val_df = Train_Val_df.generate_df_from_json(json_data)

    # train_data = train_df[:100]
    # dev_data = val_df[:100]


    # DatasetDict creation
    dataset_our = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(val_df)
    })

    # Cast audio column
    dataset_our = dataset_our.cast_column("audio", Audio(sampling_rate=16000,  mono=True))

    # Functions
    # ... (Rest of your code, including functions like prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding, etc.)
    


    # Prepare dataset
    # mp.set_start_method("spawn", force=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # dataset_our["train"] = dataset_our["train"].map(lambda batch: prepare_dataset_gpu(batch, device), num_proc=2)
    # dataset_our["test"] = dataset_our["test"].map(lambda batch: prepare_dataset_gpu(batch, device), num_proc=2)


    # def prepare_dataset_parallel(dataset, device, num_workers=4):
    #     with mp.Pool(processes=num_workers) as pool:
    #         results = pool.map(lambda batch: prepare_dataset_gpu(batch, device), dataset)
    #     return results

    # def prepare_dataset_parallel(dataset, device):
    #     with torch.multiprocessing.Pool() as pool:
    #         func = functools.partial(prepare_dataset_gpu, device=device)
    #         results = pool.map(func, dataset)
    #     return results

    def prepare_dataset_parallel(dataset, device, batch_size=100, num_workers=2):
        num_batches = math.ceil(len(dataset) / batch_size)

        results = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(dataset))
            batch = dataset[start:end]

            print(f"Batch {batch_idx}: {batch.keys()}")

            # Create a list of dictionaries with 'audio' and 'sentence' keys
            formatted_batch = [{'audio': audio_element, 'sentence': sentence}
                            for audio_element, sentence in zip(batch['audio'], batch['sentence'])]

            with torch.multiprocessing.Pool(processes=num_workers) as pool:
                func = functools.partial(prepare_dataset, device=device)
                print("First element in the formatted batch:", formatted_batch[0])  # Update the print statement
                batch_results = pool.map(func, formatted_batch)

            results.extend(batch_results)

        return results





    # Prepare dataset
    train_data_prepared = prepare_dataset_parallel(dataset_our["train"], device)
    test_data_prepared = prepare_dataset_parallel(dataset_our["test"], device)

    # dataset_our["train"] = dataset_our["train"].map(lambda e: train_data_prepared[e['idx']], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda e: test_data_prepared[e['idx']], with_indices=True)

    # dataset_our["train"] = dataset_our["train"].map(lambda i, e: train_data_prepared[e['idx']], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda i, e: test_data_prepared[e['idx']], with_indices=True)

    # dataset_our["train"] = dataset_our["train"].map(lambda i, e: train_data_prepared[i], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda i, e: test_data_prepared[i], with_indices=True)

    # dataset_our["train"] = dataset_our["train"].map(lambda i, e: train_data_prepared[int(i)], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda i, e: test_data_prepared[int(i)], with_indices=True)

    # dataset_our["train"] = dataset_our["train"].map(lambda i, e: train_data_prepared[int(e['idx'])], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda i, e: test_data_prepared[int(e['idx'])], with_indices=True)

    # dataset_our["train"] = dataset_our["train"].map(lambda i, e: train_data_prepared[int(e.get('idx'))], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda i, e: test_data_prepared[int(e.get('idx'))], with_indices=True)

    dataset_our["train"] = dataset_our["train"].map(
        lambda i, e: train_data_prepared[int(e.get('idx'))] if isinstance(e, dict) else train_data_prepared[int(e)],
        with_indices=True
    )

    dataset_our["test"] = dataset_our["test"].map(
        lambda i, e: test_data_prepared[int(e.get('idx'))] if isinstance(e, dict) else test_data_prepared[int(e)],
        with_indices=True
    )



    def is_not_skipped(batch):
        return batch is not None

    dataset_our["train"] = dataset_our["train"].filter(is_not_skipped, num_proc=NUM_PROC)
    dataset_our["test"] = dataset_our["test"].filter(is_not_skipped, num_proc=NUM_PROC)



  

    # dataset_our["train"] = dataset_our["train"].map(lambda i, e: train_data_prepared[i], with_indices=True)
    # dataset_our["test"] = dataset_our["test"].map(lambda i, e: test_data_prepared[i], with_indices=True)


    # Dataset processing
    # ... (Rest of your code, including dataset processing and filtering)

    # Data collator
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, forward_attention_mask=apply_spec_augment)

    # Compute metrics
    wer_metric = evaluate.load("wer", cache_dir=os.path.join(base_dir, "metrics_cache"))
    cer_metric = evaluate.load("cer", cache_dir=os.path.join(base_dir, "metrics_cache"))
    do_normalize_eval = False

    def compute_metrics(pred):
        # ... (Rest of your code, including the compute_metrics function implementation)
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]

        # wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        # cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}

    # Load model
    print("\n\n Loading Model to Device..\n\n")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    ## Override generation arguments
    model.config.apply_spec_augment = apply_spec_augment
    model.config.dropout = dropout
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if gradient_checkpointing:
        model.config.use_cache = False
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    
    ## Truncate_model
    def truncate_model(model, d_layers_to_remove=0, e_layers_to_remove=0):
        print(f"e_layers_to_remove {e_layers_to_remove}, d_layers_to_remove {d_layers_to_remove}")
        num_e_layers = len(model.model.encoder.layers)
        num_d_layers = len(model.model.decoder.layers)
        model_truncated = copy.deepcopy(model)
        model_truncated.model.encoder.layers = torch.nn.ModuleList(list(model.model.encoder.layers.children()))[:num_e_layers-e_layers_to_remove]
        model_truncated.model.decoder.layers = torch.nn.ModuleList(list(model.model.decoder.layers.children()))[:num_d_layers-d_layers_to_remove]
        return model_truncated
        
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoint/whisper-small-900hr-pre-processing-gpu",
        # overwrite_output_dir=overwrite_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        bf16=bf16,
        tf32=tf32,
        generation_max_length=generation_max_length,
        # report_to=report_to,
        predict_with_generate=predict_with_generate,
        push_to_hub=push_to_hub,
        hub_token="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",
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_our["train"],
        eval_dataset=dataset_our["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        # callbacks=[tf.EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Train the model
    print("\n\n Training the model...\n\n")
    train_result = trainer.train()
    print("\n\n Training completed.\n\n")

    # Save the model
    print("\n\n Saving the trained model...\n\n")
    trainer.save_model(output_dir)
    print(f"\n\n Model saved in {output_dir}.\n\n")

    # Evaluate the model
    print("\n\n Evaluating the model...\n\n")
    eval_result = trainer.evaluate()
    print(f"\n\n Evaluation results: {eval_result}\n\n")

    return eval_result


if __name__ == "__main__":
    mp.set_start_method('spawn')
    res = main()
