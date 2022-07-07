""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols_ko import symbols_ko
from text.symbols_en import symbols_en

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id_ko = {s: i for i, s in enumerate(symbols_ko)}
_id_to_symbol_ko = {i: s for i, s in enumerate(symbols_ko)}

_symbol_to_id_en = {s: i for i, s in enumerate(symbols_en)}
_id_to_symbol_en = {i: s for i, s in enumerate(symbols_en)}

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)

  if cleaner_names[0] == "english_cleaners":
    for symbol in clean_text:
      try:
        symbol_id = _symbol_to_id_en[symbol]
        sequence += [symbol_id]
      except:
        print(_symbol_to_id_en)
        print("WARNING : Skipping en token, " + symbol)  
  else:
    for symbol in clean_text:
      try:
        symbol_id = _symbol_to_id_ko[symbol]
        sequence += [symbol_id]
      except:
        print("WARNING : Skipping ko token, " + symbol)
    
  return sequence


def cleaned_text_to_sequence(cleaned_text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if cleaner_names[0] == "english_cleaners":
    sequence = [_symbol_to_id_en[symbol] for symbol in cleaned_text]
  else:
    sequence = [_symbol_to_id_ko[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence, cleaner_names):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if cleaner_names[0] == "english_cleaners":
      s = _id_to_symbol_en[symbol_id]
    else:
      s = _id_to_symbol_ko[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
