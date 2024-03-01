'''
Vietnamese ASR with Whisper-small
'''
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

max_input_length = 30.0
def is_audio_in_length_range(length):
    return length < max_input_length

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"]
    )
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


# WER of pretrained whisper
if torch.cuda.is_available():
  device = "cuda:0"
  torch_dtype = torch.float16
  print("using GPU")
else:
  device = "cpu"
  torch_dtype = torch.float32
  print("using CPU")

"""**Finetune Whisper-Small for Vietnamese ASR**"""
# Baseline: whisper-small on vietnamese test on common voice 13 dataset gives a wer of >43% (0.43239669421487603) (without removing case or puncs) and >34% (0.34472118139739916) (after removing case or puncs)

ds = DatasetDict()
ds["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="train+validation")
ds["test"]  = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="test")
# ds features: 'client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'

ds = ds.select_columns(["audio", "sentence"])
# ds features: "audio", "sentence"?
print("after select columns:")
print(ds)


"""ASR pipeline:
1. Feature extractor: wav2mel:  preprocesses raw audio to log-mel spectrograms
2. Run the features through whisper
3. Tokenizer (Token2text)
Feature extractor and Tokenizer are included in WhisperProcessor. Or can call associated feature extractor and tokenizer, called WhisperFeatureExtractor and WhisperTokenizer respectively

1. Feature extractor
"""
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vietnamese", task="transcribe")

# Preprocess data

sampling_rate = processor.feature_extractor.sampling_rate

# This operation does not change the audio in-place, but rather signals to datasets to resample audio samples on-the-fly when they are loaded
print('Before cast_column:')
print(ds["train"].features)
ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
print('After cast_column:')
print(ds["train"].features)

# Func to preprocess data
# 1. Load & resample on sample-by-sample basis by calling sample["audio"] : Datasets performs resampling on the fly
# 2. wav2mel
# 3. token2text


# Apply above func to all training examples
print("Mapping: ")
print("removing columns: ")
print(ds.column_names["train"])
ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)

# Filter out exxamples over 30s
ds["train"] = ds["train"].filter(
      is_audio_in_length_range,
      input_columns=["input_length"]
)
# ds: 'input_features', 'labels', 'input_length'
print("current dataset:")
print(ds)

"""**Training and Evaluation**"""
"""
What are needed:
1. Define a data collator which takes our pre-processed data and prepares PyTorch tensors ready for the model
2. Evaluation metrics: during evaluation, we want to evaluate the model using the word error rate (WER) metric. We need to define a compute_metrics function that handles this computation.
3. Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.
4. Define the training arguments: these will be used by Trainer in constructing the training schedule.
"""

"""
1. Define a data collator: takes the pre-processed data and prepares PyTorch tensors ready for the model
"""
# What's happen in the following function:
# convert input_features, which are log-mels of 30s padded audio, to batched pytorch tensors
# Pad labels (sentences) to max length in the batch. Cut SOT token

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

"""
2. Evaluation metrics: during evaluation, we want to evaluate the model using the word error rate (WER) metric. We need to define a compute_metrics function that handles this computation.
"""
metric = evaluate.load("wer")

# Define func that maps model predictions to wer

normalizer = BasicTextNormalizer()

# Take model prediction and return wer
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


"""
3. Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.
"""
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="vietnamese", task="transcribe", use_cache=True
)

"""
4. Define the training arguments: these will be used by the 🤗 Trainer in constructing the training schedule.
"""
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-vi",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=2e-5,
    lr_scheduler_type="linear",# set to linear" if set max_steps large (~4000) "as this will yield an additional performance boost over long training runs".
    warmup_steps=50,
    max_steps=1500,#500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    #save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    #push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()