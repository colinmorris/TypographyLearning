from optparse import OptionParser
import os
import numpy as np
import random

# Bleh. TODO
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import common

def get_ngrams(tokens, max_size, n):
  count = 0
  acc = ''
  for word in tokens:
    if count >= n:
      break
    if not all(c == ' ' or 97 <= ord(c) <= 122 for c in word):
      continue
    if len(word) > max_size:
      continue
    if len(word) + len(acc) + 1 > max_size:
      yield acc
      count +=1
      acc = word
    else:
      acc += (' ' if acc else '') + word

def random_token_stream():
  min_token = 1
  max_token = 12
  while 1:
    token_len = np.clip(int(random.gauss(5,2)), min_token, max_token)
    yield ''.join(chr(random.randint(ord('a'), ord('z'))) for _ in range(token_len))

		

parser = OptionParser()
parser.add_option('-n', dest='n', default=10)
parser.add_option('-d', dest='out', default='imgs', help='output directory for images')

(options, args) = parser.parse_args()

if len(args) < 1:
  print "WARNING: Didn't get an input file, so using randomly generated text"
  tokens = random_token_stream()
else:
  f = open(args[0])
  text = f.read()
  tokens = text.lower().split()

seen = set()
for ngram in get_ngrams(tokens, 12, int(options.n)):
  if ngram in seen:
    continue
  seen.add(ngram)
  # Keep it easy for now
  if len(ngram) > 12:
    continue
  fname = options.out + '/' + ngram.replace(' ', '_') + '.pgm'
  os.system("convert -background white -fill black -pointsize 14 -font 'DejaVu-Sans-Mono-Book' -size {}x{} -gravity West -depth 8 caption:'{}' {}".format(common.IMG_WIDTH, common.IMG_HEIGHT, ngram, fname))
