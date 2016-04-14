from optparse import OptionParser
import os
from .. import common

def get_ngrams(text, max_size, n):
  # This is kind of slow probably
	words = text.lower().split()
	count = 0
	acc = ''
	for word in words:
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
		

parser = OptionParser()
parser.add_option('-n', dest='n', default=10)
parser.add_option('-d', dest='out', default='imgs', help='output directory for images')

(options, args) = parser.parse_args()

f = open(args[0])
text = f.read()

seen = set()
for ngram in get_ngrams(text, 12, int(options.n)):
	if ngram in seen:
		continue
	seen.add(ngram)
	# Keep it easy for now
	if len(ngram) > 12:
		continue
	fname = options.out + '/' + ngram.replace(' ', '_') + '.pgm'
	os.system("convert -background white -fill black -pointsize 14 -font 'DejaVu-Sans-Mono-Book' -size {}x{} -gravity West -depth 8 caption:'{}' {}".format(common.IMG_WIDTH, common.IMG_HEIGHT, ngram, fname))
