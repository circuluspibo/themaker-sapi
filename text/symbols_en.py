""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''

from text import cmudict
from .korean import KOR_SYMBOLS

kor_symbols=KOR_SYMBOLS
#symbols=kor_symbols


_pad        = '_'
_eos        = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet


_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
#_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols_en = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    #+ list(_letters)
    + cmudict.valid_symbols#_arpabet
)