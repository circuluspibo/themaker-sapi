""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from phonemizer import phonemize
import jamotools

import re
from jamo import h2j

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

en2kr = {
	"A" : "에이",
	"B" : "비이",
	"C" : "씨이",
	"D" : "디이",
	"E" : "이",
	"F" : "에프",
	"G" : "쥐이",
	"H" : "에이취",
	"I" : "아이",
	"J" : "제이",
	"K" : "케이",
	"L" : "엘",
	"M" : "엠",
	"N" : "엔",
	"O" : "오",
	"P" : "피이",
	"Q" : "큐우",
	"R" : "아르",
	"S" : "에스",
	"T" : "티이",
	"U" : "유",
	"V" : "브이",
	"W" : "떠블류",
	"X" : "엑스",
	"Y" : "와이",
	"Z" : "지이"
}


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

'''
def english_cleaners(text):
  
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes
'''

def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def korean_jamo_cleaners(text):
  '''Pipeline for Korean text. Split Korean into Jamo syllables.'''

  text = jamotools.split_syllables(text, jamo_type="JAMO")
  text = text.replace('@', '')
  return text

from g2pk import G2p
g2pK = G2p()

from g2p_en import G2p
g2pE = G2p()

#def korean_cleaners(text):
#    text=ko_tokenize(text, as_id=False)
#    return text 

#without h2j
def korean_cleaners(text):

  # 20.2 -> 20 jum 2
  list = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", text)

  for num in list:
    if num.find(".") != -1:
      text = text.replace(num,num.replace('.','점'))

  text = g2pK(text.strip(), descriptive=True, group_vowels=True)
  text = jamotools.split_syllables(text, jamo_type="JAMO")
  text = text.replace('@', '').replace("\"","").replace("‘","").replace("’","").replace('”',"").replace("…","")

  result = re.compile('[a-zA-Z]').findall(text) # [a-z|A-Z]
  #print(result)
  for ch in result:
	  text = text.replace(ch, en2kr[ch.upper()])

  text = h2j(text)
  #print(text)
  return text
  
def english_cleaners(text):
  text = lowercase(text)
  text = expand_abbreviations(text)
  text = g2pE(text.strip())
  #phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  #phonemes = collapse_whitespace(phonemes)
  return text  
