import logging
from fastapi import FastAPI, File, UploadFile
logging.basicConfig(level=logging.ERROR) #instead of Level=logging.DEBUG

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols_ko import symbols_ko
from text.symbols_en import symbols_en
from text import text_to_sequence
import soundfile as sf
import numpy as np
#import librosa
from pydantic import BaseModel
import sox
from pydub import AudioSegment
#from kss import split_sentences

from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from lxml.etree import fromstring
#import pydub
import time
import numpy

#import pysbd # english
import re
import sys
import hashlib

class Param (BaseModel):
    text : str
    hash = ""
    voice = "main" 
    lang = "ko"
    type ="mp3"


app = FastAPI()
hps_ko = utils.get_hparams_from_file("./configs/main_base.json")
hps_en = utils.get_hparams_from_file("./configs/kevin_base.json")

g_main = SynthesizerTrn( len(symbols_ko),hps_ko.data.filter_length // 2 + 1, hps_ko.train.segment_size // hps_ko.data.hop_length, **hps_ko.model).cuda()
g_main.eval()
utils.load_checkpoint("./logs/main/last.pth", g_main, None)

g_boy = SynthesizerTrn( len(symbols_ko),hps_ko.data.filter_length // 2 + 1, hps_ko.train.segment_size // hps_ko.data.hop_length, **hps_ko.model).cuda()
g_boy.eval()
utils.load_checkpoint("./logs/boy/last.pth", g_boy, None)

g_girl = SynthesizerTrn( len(symbols_ko),hps_ko.data.filter_length // 2 + 1, hps_ko.train.segment_size // hps_ko.data.hop_length, **hps_ko.model).cuda()
g_girl.eval()
utils.load_checkpoint("./logs/girl/last.pth", g_girl, None)

g_man1 = SynthesizerTrn( len(symbols_ko),hps_ko.data.filter_length // 2 + 1, hps_ko.train.segment_size // hps_ko.data.hop_length, **hps_ko.model).cuda()
g_man1.eval()
utils.load_checkpoint("./logs/man1/last.pth", g_man1, None)

g_woman1 = SynthesizerTrn( len(symbols_ko),hps_ko.data.filter_length // 2 + 1, hps_ko.train.segment_size // hps_ko.data.hop_length, **hps_ko.model).cuda()
g_woman1.eval()
utils.load_checkpoint("./logs/woman1/last.pth", g_woman1, None)

# english suport
e_main = SynthesizerTrn( len(symbols_en),hps_en.data.filter_length // 2 + 1, hps_en.train.segment_size // hps_en.data.hop_length, **hps_en.model).cuda()
e_main.eval()
utils.load_checkpoint("./logs/e_main/last.pth", e_main, None)

e_boy = SynthesizerTrn( len(symbols_en),hps_en.data.filter_length // 2 + 1, hps_en.train.segment_size // hps_en.data.hop_length, **hps_en.model).cuda()
e_boy.eval()
utils.load_checkpoint("./logs/e_boy/last.pth", e_boy, None)

e_girl = SynthesizerTrn( len(symbols_en),hps_en.data.filter_length // 2 + 1, hps_en.train.segment_size // hps_en.data.hop_length, **hps_en.model).cuda()
e_girl.eval()
utils.load_checkpoint("./logs/e_girl/last.pth", e_girl, None)

e_man1 = SynthesizerTrn( len(symbols_en),hps_en.data.filter_length // 2 + 1, hps_en.train.segment_size // hps_en.data.hop_length, **hps_en.model).cuda()
e_man1.eval()
utils.load_checkpoint("./logs/e_man1/last.pth", e_man1, None)

e_woman1 = SynthesizerTrn( len(symbols_en),hps_en.data.filter_length // 2 + 1, hps_en.train.segment_size // hps_en.data.hop_length, **hps_en.model).cuda()
e_woman1.eval()
utils.load_checkpoint("./logs/e_woman1/last.pth", e_woman1, None)

device = "cuda:0"

#from speechbrain.pretrained import EncoderDecoderASR
from transformers import AutoProcessor, AutoModelForCTC
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

#asr_model = EncoderDecoderASR.from_hparams("speechbrain/asr-conformer-transformerlm-ksponspeech",run_opts={"device":device})

ko_processor = AutoProcessor.from_pretrained("cheulyop/wav2vec2-large-xlsr-ksponspeech_1-20")
ko_model = AutoModelForCTC.from_pretrained("cheulyop/wav2vec2-large-xlsr-ksponspeech_1-20")
ko_model = ko_model.to(device)
 
 # load model and processor
en_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
en_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


try:
    ENV = os.environ['CIRCULUS_ENV']
except:
    ENV = "DEV"

conf = json.load(open('config.json', 'r'))
CONF = conf[ENV]
print(CONF)

print("CIRCULUS_ENV: " + ENV)

def get_text(text, voice):
    if  voice.startswith('e_'):
        text_norm = text_to_sequence(text, hps_en.data.text_cleaners)
        if hps_en.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
    else:
        text_norm = text_to_sequence(text, hps_ko.data.text_cleaners)
        if hps_ko.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)

    return torch.LongTensor(text_norm)

def inference(text, voice="main", lang="ko"):

    if lang == "en":
        voice = "e_" + voice

    stn_tst = get_text(text, voice)

    #print(voice)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

        if voice == "main":
            audio = g_main.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "girl":
            audio = g_girl.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "boy":
            audio = g_boy.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "man1":
            audio = g_man1.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "woman1":
            audio = g_woman1.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "e_main":
            audio = e_main.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "e_girl":
            audio = e_girl.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "e_boy":
            audio = e_boy.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "e_man1":
            audio = e_man1.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif voice == "e_woman1":
            audio = e_woman1.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    return audio

def clean(raw):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', str(raw))
    return cleantext

#break @strength none x-weak weak medium strong x-strong
#sub @alias
#prosody 
# @pitch x-low low medium high x-high  
# @rate x-slow slow medium fast x-fast 
# @volume silent x-soft soft medium loud x-loud
#emphasis 
# @level strong moderate reduced none
#voice  @name
#mark @name

def parseEmphasis(o="morderate"):
    if o == "none":
        return 0
    elif o == "reduced":
        return 1.05
    elif o == "strong":
        return 1.15
    else: #morderate
        return 1.1

def parseBreak(o="medium"):
    if o == "none":
        return 0
    elif o == "x-weak":
        return 5000
    elif o == "weak":
        return 7500  
    elif o == "strong":
        return 30000  
    elif o == "x-strong":
        return 60000    
    else: #medium
        return 15000

def parsePitch(o="medium"): #tone
    if o == "x-high":
        return 1.5
    elif o == "high":
        return 0.75
    elif o == "low":
        return -0.75
    elif o == "x-low":
        return -1.5            
    else: #medium
        return 0

def parseRate(o="medium"): #speed
    if o == "x-fast":
        return 1.2
    elif o == "fast":
        return 1.1
    elif o == "slow":
        return 0.9
    elif o == "x-slow":
        return 0.8              
    else: #medium
        return 1

def parseVolume(o="medium"): 
    if o == "silent":
        return 0.2
    elif o == "x-soft":
        return 0.8
    elif o == "soft":
        return 0.9
    elif o == "loud":
        return 1.1
    elif o == "x-loud":
        return 1.2             
    else: #medium
        return 1        

def inferenceSSML(text, voice, lang="ko"):
    wavs = []
    items = xmlParser(text)

    for item in items:
        #print(voice, item)
        if "k" not in item:
            if "n" in item:
                wav = inference(item["v"], item["n"],lang)
            else:
                wav = inference(item["v"], voice,lang)
            if "p" in item:
                tfm = sox.Transformer()
                if "pitch" in item["p"]:
                    tfm.pitch(parsePitch(item["p"]["pitch"]))
                if "rate" in item["p"]:
                    tfm.tempo(parseRate(item["p"]["rate"]))
                if "volume" in item["p"]:
                    tfm.gain(parseVolume(item["p"]["volume"]))
                wav = tfm.build_array(input_array=wav, sample_rate_in=22050)    
            wavs += list(wav)
        elif item["k"] == "break":
            if "o" in item and "strength" in item["o"]:
                wavs += [0] * parseBreak(item["o"]["strength"])
            else:
                wavs += [0] * parseBreak()
        elif item["k"] == "sub":
            wav = inference(item["o"]["alias"], voice,lang)
            if "p" in item:
                tfm = sox.Transformer()
                if "pitch" in item["p"]:
                    tfm.pitch(parsePitch(item["p"]["pitch"]))
                if "rate" in item["p"]:
                    tfm.tempo(parseRate(item["p"]["rate"]))
                if "volume" in item["p"]:
                    tfm.gain(parseVolume(item["p"]["volume"]))
                wav = tfm.build_array(input_array=wav, sample_rate_in=22050)    
            wavs += list(wav)            

        elif item["k"] == "voice":
            wav = inference(item["v"], item["o"]["name"],lang)
            if "p" in item:
                tfm = sox.Transformer()
                if "pitch" in item["p"]:
                    tfm.pitch(parsePitch(item["p"]["pitch"]))
                if "rate" in item["p"]:
                    tfm.tempo(parseRate(item["p"]["rate"]))
                if "volume" in item["p"]:
                    tfm.gain(parseVolume(item["p"]["volume"]))
                wav = tfm.build_array(input_array=wav, sample_rate_in=22050)    
            wavs += list(wav)            
        elif item["k"] == "emphasis":
            if "n" in item:
                wav = inference(item["v"], item["n"],lang)
            else:
                wav = inference(item["v"], voice,lang)

            tfm = sox.Transformer()

            if "level" in item["o"]: 
                tfm.gain(parseEmphasis(item["o"]["level"])) #loudness 
                #wavs += list(librosa.effects.preemphasis(wav, coef=parseEmphasis(item["o"]["level"])))
            else:
                tfm.gain(parseEmphasis())
                #wavs += list(librosa.effects.preemphasis(wav, coef=parseEmphasis()))
            tfm.pitch(0.95)
            wavs += list(tfm.build_array(input_array=wav, sample_rate_in=22050))

            if "p" in item:
                tfm = sox.Transformer()
                if "pitch" in item["p"]:
                    tfm.pitch(parsePitch(item["p"]["pitch"]))
                if "rate" in item["p"]:
                    tfm.tempo(parseRate(item["p"]["rate"]))
                if "volume" in item["p"]:
                    tfm.gain(parseVolume(item["p"]["volume"]))
                wav = tfm.build_array(input_array=wav, sample_rate_in=22050)    

        elif item["k"] == "prosody":
            print("prosody", item)
            if "n" in item:
                wav = inference(item["v"], item["n"],lang)
            else:
                wav = inference(item["v"], voice,lang)

            if "o" in item:
                tfm = sox.Transformer()

                if "pitch" in item["o"]:
                    #print("pitch",item["o"]["pitch"])
                    tfm.pitch(parsePitch(item["o"]["pitch"]))
                    #wav = librosa.effects.pitch_shift(wav, sr=22050, n_steps=parsePitch(item["o"]["pitch"]))
                if "rate" in item["o"]:
                    tfm.tempo(parseRate(item["o"]["rate"]))
                    #wav = librosa.effects.time_stretch(wav, rate=parseRate(item["o"]["rate"]))
                if "volume" in item["o"]:
                    tfm.gain(parseVolume(item["o"]["volume"]))
                    #wav = librosa.util.normalize(wav, norm=parseVolume(item["o"]["volume"]), fill=True)

                wav = tfm.build_array(input_array=wav, sample_rate_in=22050)
            else:
                print("not detect")
            wavs += list(wav)
        else:
            print("not support now, " + item["k"])
            wavs += list(inference(item["v"], voice,lang))

    return wavs

def xmlParser(text):
    speaks = []
    #text = """<speak>Step 1, take a deep 숨쉬기. <break time="200ms"/>스텝 1, 베리 구웃입니다.<voice name="man1">Step 2, exhale.</voice>스텝 2, 분리가 되나요? <prosody rate="slow" pitch="-2st">Can you hear me now?</prosody><emphasis level="moderate">Step 4, exhale.</emphasis></speak>"""
    
    tree = fromstring(text)

    if tree.text != None:
        #print(tree.tag + "|"+ str(len(tree)))
        speaks.append({ "v" : clean(tree.text)})

    for item in tree:
        #print(item.tag + "|" +  str(len(item)))
        #print(item.attrib)
        #print(item.text)
        #print(item.tail)

        if item.tag == "break":
            speaks.append({ "k" : item.tag, "o" : item.attrib })
            if item.tail != None:
                speaks.append({ "v" : clean(item.tail)})
        else: #no text in tag
            if item.text != None:
                speaks.append({ "k" : item.tag, "v" : clean(item.text) , "o" : item.attrib })
            if item.tail != None:
                print(item.tail)
                speaks.append({ "v" : clean(item.tail)})

        if len(item) > 0:
            if item.tag == "voice":
                for sub in item:
                    #print("-" + sub.tag  + "|" +  str(len(sub)))
                    if sub.tag == "break": #sub.text != None and 
                        speaks.append({ "k" : "break", "o" : sub.attrib, "n" : item.get("name") })
                        if sub.tail != None:
                            speaks.append({ "v" : clean(sub.tail) , "n" : item.get("name")})
                    else:
                        if sub.text != None:
                            speaks.append({ "k" : sub.tag, "v" : clean(sub.text) , "o" : sub.attrib, "n" : item.get("name") })
                        if sub.tail != None:
                            speaks.append({ "v" : clean(sub.tail), "n" : item.get("name")}) 
            elif item.tag == "prosody":
                for sub in item:
                    if sub.tag == "break": #sub.text != None and 
                        speaks.append({ "k" : "break", "o" : sub.attrib, "p" : item.attrib })
                        if sub.tail != None:
                            speaks.append({ "v" : clean(sub.tail) , "p" : item.attrib})
                    else:
                        if sub.text != None:
                            speaks.append({ "k" : sub.tag, "v" : clean(sub.text) , "o" : sub.attrib, "p" : item.attrib })
                        if sub.tail != None:
                            speaks.append({ "v" : clean(sub.tail), "p" : item.attrib}) 
            else:
                for sub in item:
                    #print(sub.tag + "|" +  str(len(sub)))
                    if sub.text != None and sub.tag == "break":
                        speaks.append({ "k" : "break", "o" : sub.attrib })
                        if sub.tail != None:
                            speaks.append({ "v" : clean(sub.tail)})
                    else:
                        speaks.append({ "k" : sub.tag, "v" : clean(sub.text) , "o" : sub.attrib })
                        if sub.tail != None:
                            speaks.append({ "v" : clean(sub.tail)})            

    return speaks


def parser(text,voice="main",lang="ko"):
    wavs = []
    sens = clean(text).split(". ")#sbd.segment(text)
    #sens = split_sentences(text) # performance is bad
    print(sens)

    for sen in sens:
        if len(sen) == 1:
            continue
        wav = inference(sen+".", voice,lang)
        wavs += list(wav)
        wavs += [0] * 7500

    return wavs

@app.get("/")
async def main():
	return { "result" : True, "data" : "CIRCULUS-TAPI-V2"}

"""
def save(f, x, normalized=False):
    #numpy array to MP3
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=22050, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
"""
#sbd = pysbd.Segmenter(language="en", clean=True)

# 127.0.0.1:57727/tts?text='<speak>오늘은 뭘 해보면 좋을까?<break strength="x-strong"/>프로그래밍을 해보자. <voice name="man1">프로그램은 정말로 좋은 것입니다.</voice>정말요? <prosody rate="slow" pitch="x-high">놀라움의 연속이군요!</prosody> 잘해볼께요!</speak>'



@app.post("/tts", response_class=FileResponse)
async def tts2(param : Param): #ogg , flac, wav
    print(param)
    text = param.text
    voice = param.voice
    lang = param.lang
    type = param.type  #ogg , flac, wav
    
    start = time.time()  # 시작 시간 저장

    if param.hash != "":
        hash = param.hash
    else:
        hash = hashlib.md5((text +"_"+ voice +"_"+ lang + "_" + str(conf['VER'])).encode('utf-8')).hexdigest()    

    print(hash)

    wav = "speech/" + hash + ".wav"
    file = "speech/" + hash + "." + type
    
    wavs = []

    if os.path.isfile(file):
        print("Cached:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        return file
    elif "<speak>" in text:
        try:
            wavs = inferenceSSML(text, voice, lang)
        except Exception as e:
            print(e)
            wavs = parser(text , voice, lang)
        print("Inference SSML:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간            
    else:
        wavs = parser(text , voice, lang)
        print("Inference plain:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간            
    
    sf.write(wav, wavs, 22050)

    if type == "flac":
        sf.write(file, wavs, 22050)
    elif type == "mp3":        
        sound = AudioSegment.from_wav(wav)
        sound.export(file, format="mp3", bitrate="64k")
        
    print("Generated:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    return file

@app.get("/tts", response_class=FileResponse)
async def tts(text : str, hash="", voice = "main", lang = "ko", type="mp3"): #ogg , flac, wav
    start = time.time()  # 시작 시간 저장

    if hash == "":
        hash = hashlib.md5((text +"_"+ voice +"_"+ lang + "_" + str(conf['VER'])).encode('utf-8')).hexdigest()    
    wav = "speech/" + hash + ".wav"
    file = "speech/" + hash + "." + type
    
    print(hash)

    wavs = []

    if os.path.isfile(file):
        print("Cached:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        return file
    elif "<speak>" in text:
        try:
            wavs = inferenceSSML(text, voice, lang)
        except Exception as e:
            print(e)
            wavs = parser(text , voice, lang)
        print("Inference SSML:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간            
    else:
        wavs = parser(text , voice, lang)
        print("Inference plain:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간            
    
    sf.write(wav, wavs, 22050)

    if type == "flac":
        sf.write(file, wavs, 22050)
    elif type == "mp3":        
        sound = AudioSegment.from_wav(wav)
        sound.export(file, format="mp3", bitrate="64k")
        
    print("Generated:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    return file

@app.post("/stt")
async def stt(file : UploadFile = File(...), lang = "ko"):
    start = time.time()  # 시작 시간 저장
    location = f"files/{file.filename}"

    with open(location,"wb+") as file_object:
        file_object.write(file.file.read())
    #if lang == "ko":
    #    transcription = asr_model.transcribe_file(location)
    #    print("Transcript1:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    #    return { "result" : True, "data" : transcription}
    #elif lang == "ko2":
        if lang == "ko":
            speech_array, sampling_rate = torchaudio.load(location)
            speech_array.to(device)

            feat =ko_processor(speech_array[0], sampling_rate=16000, padding=True, max_length=800000, 
            truncation=True, return_attention_mask=True, return_tensors="pt", pad_token_id=49)

            feat.to(device)
            input = {'input_values': feat['input_values'],'attention_mask':feat['attention_mask']}
            #input.to(device)
            outputs = ko_model(**input, output_attentions=True)
            logits = outputs.logits
            predicted_ids = logits.argmax(axis=-1)
            transcription = ko_processor.decode(predicted_ids[0])
            print("Transcript 1:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
            return { "result" : True, "data" : transcription}
        else:
            speech_array, sampling_rate = torchaudio.load(location)
            speech_array.to(device)
            input_values = en_processor(speech_array[0], return_tensors="pt", padding="longest").input_values
            
            # retrieve logits
            logits = en_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = en_processor.batch_decode(predicted_ids)
            print("Transcript 2:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
            return { "result" : True, "data" : transcription}


if __name__ == "__main__":
    uvicorn.run("index:app",host=CONF["HOST"],port=CONF["PORT"])
