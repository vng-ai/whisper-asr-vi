"""
Do inference
"""
from transformers import pipeline
import torch
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset


# WER of pretrained whisper
if torch.cuda.is_available():
  device = "cuda:0"
  torch_dtype = torch.float16
  print("using GPU")
else:
  device = "cpu"
  torch_dtype = torch.float32
  print("using CPU")


ds_test = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="test")
pipe = pipeline("automatic-speech-recognition",
            model="openai/whisper-small",
            torch_dtype=torch_dtype,
            device=device,
            )

# v = ds_test['audio']
# vv = v[0:2]
# outs = pipe(vv)
# outs
# print(ds_test[0]['sentence'])
# print(ds_test[1]['sentence'])

all_predictions = []

# whisper takes in list of "audio" values
# run streamed inference
for prediction in tqdm(
pipe(
    KeyDataset(ds_test, "audio"),
    max_new_tokens=128,
    generate_kwargs={"task": "transcribe"},
    batch_size=32,
),
desc="Testing Whisper",
total=len(ds_test),
):
all_predictions.append(prediction["text"])

all_predictions[:10]

# # Alternative implementation:
# audios = ds_test[:10]["audio"]
# idx=0
# all_predictions_=[]
# for prediction_ in tqdm(
#     pipe(
#         audios,
#         max_new_tokens=128,
#         generate_kwargs={"task": "transcribe"},
#         batch_size=32,
#     ),
#     desc="Testing Whisper",
#     total=len(audios),
# ):
#     #print("\n{}".format(prediction_["text"]))
#     assert prediction_["text"]==all_predictions[idx]
#     idx+=1
#     all_predictions_.append(prediction_["text"])

# Uncomment this for santiy check
#v = ds_test['audio']# list
#key_ds = KeyDataset(ds_test, "audio")
#import numpy as np
#for idx in range(min(10, len(v))):
#    assert np.all(key_ds[idx]['array']==v[idx]["array"])
#print("KeyDataset and Van's way to create list of 'audio' colums yield identical arrays")

#Compute WER without removing case or punctuations
wer_metric = load("wer")
wer_ortho = wer_metric.compute(references=ds_test["sentence"], predictions=all_predictions)

#Compute WER after removing case or punctuations
normalizer = BasicTextNormalizer()# to remove case & puntuations
ref_normalized = [normalizer(item) for item in ds_test["sentence"]]
pred_normalized = [normalizer(item) for item in all_predictions]
# Remove empty sentences
ref_normalized = [item for item in ref_normalized if len(item)>0]
pred_normalized = [item for item in pred_normalized if len(item)>0]

wer = wer_metric.compute(references=ref_normalized, predictions=pred_normalized)
