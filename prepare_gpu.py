from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast, WhisperProcessor
model_name = "openai/whisper-tiny"
import torch
import librosa
# model_name = "openai/whisper-small"
# model_name = "openai/whisper-large-v2"

language = "Bengali"
task = "transcribe"
apply_spec_augment = True


# def prepare_dataset_gpu(batch, device):
#     # load and (possibly) resample audio data to 16kHz
#     audio = batch["audio"]
#     processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
#     # compute log-Mel input features from input audio array 
#     inputs = processor.feature_extractor(
#         audio["array"], 
#         sampling_rate=audio["sampling_rate"], 
#         return_attention_mask=apply_spec_augment,
#         )
    
#     input_features_tensor = torch.tensor(inputs.input_features[0]).to(device)
#     batch["input_features"] = input_features_tensor

#     # compute input length
#     batch["input_length"] = len(batch["audio"])
    
#     # if spec augmentation applied, get attention_mask to guide the mask along time axis
#     if apply_spec_augment:
#         attention_mask_tensor = torch.tensor(inputs.get("attention_mask")[0]).to(device)
#         batch["attention_mask"] = attention_mask_tensor
    
#     transcription = batch["sentence"]
    
#     # encode target text to label ids
#     encoded_transcription = processor.tokenizer(transcription).input_ids
#     labels_tensor = torch.tensor(encoded_transcription).to(device)
#     batch["labels"] = labels_tensor
    
#     # compute labels length **with** special tokens! -> total label length
#     batch["labels_length"] = len(batch["labels"])
    
#     return batch


def prepare_dataset(element, device):
    try:
        # load and (possibly) resample audio data to 16kHz
        audio = element["audio"]
        audio["array"] = librosa.util.normalize(audio["array"])
        processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
        # compute log-Mel input features from input audio array 
        inputs = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_attention_mask=apply_spec_augment,
            )
        
        input_features_tensor = torch.tensor(inputs.input_features[0]).to(device)
        element["input_features"] = input_features_tensor

        # compute input length
        element["input_length"] = len(element["audio"])
        
        # if spec augmentation applied, get attention_mask to guide the mask along time axis
        if apply_spec_augment:
            attention_mask_tensor = torch.tensor(inputs.get("attention_mask")[0]).to(device)
            element["attention_mask"] = attention_mask_tensor
        
        transcription = element["sentence"]
        
        # encode target text to label ids
        encoded_transcription = processor.tokenizer(transcription).input_ids
        labels_tensor = torch.tensor(encoded_transcription).to(device)
        element["labels"] = labels_tensor
        
        # compute labels length **with** special tokens! -> total label length
        element["labels_length"] = len(element["labels"])
        
        return element

    except Exception as e:
        print(f"Skipping this batch due to an exception: {e}")
        return None



# def prepare_dataset(batch):
#     try:
#         # load and (possibly) resample audio data to 16kHz
#         audio = batch["audio"] 
        

#         # print(audio)

#         audio["array"] = librosa.util.normalize(audio["array"])
        
#         # sf.write(f'/home/asif/whisper_900_hr_dataset/normalized_audio/{audio["path"].split("/")[-1].split(".")[0]}.flac', audio["array"], audio["sampling_rate"], format='flac')

#         # assert False
        
#         # print("test here------------------------------------")
#         # print(audio)

#         # assert False

#         # np_audio = audio.cpu().detach().numpy()

#         # audio = torch.from_numpy(librosa.util.normalize(np_audio))

#         # print("here", audio)

#         # audio_segment = pydub.AudioSegment(
#         #     audio1.tobytes(), 
#         #     frame_rate=audio["sampling_rate"],
#         #     sample_width=audio1.dtype.itemsize, 
#         #     channels=1
#         # )

#         # audio = torch.from_numpy(audio_segment)

#         # compute log-Mel input features from input audio array 
#         processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
#         inputs = processor.feature_extractor(
#             audio["array"], 
#             sampling_rate=audio["sampling_rate"], 
#             return_attention_mask=apply_spec_augment,
#             )
#         batch["input_features"] = inputs.input_features[0]


#         # print(len(batch["audio"]["array"]))
#         # compute input length
#         batch["input_length"] = len(batch["audio"])
        
#         # if spec augmentation applied, get attention_mask to guide the mask along time axis
#         if apply_spec_augment:
#             batch["attention_mask"] = inputs.get("attention_mask")[0]
        
        
#         # print(batch["sentence"])
#         # optional pre-processing steps
#         transcription = batch["sentence"]
        
        

#         # try:
#             # encode target text to label ids
#         batch["labels"] = processor.tokenizer(transcription).input_ids
#         # except:
#         #     print(transcription)
#         #     print(len(transcription))
#         #     pass
        
#         # compute labels length **with** special tokens! -> total label length
#         batch["labels_length"] = len(batch["labels"])
        
#         return batch
    
#     except Exception as e:
#         print(f"Skipping this batch due to an exception: {e}")
#         return None