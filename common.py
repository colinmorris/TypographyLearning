

TEXT_LENGTH = 12
ALPHABET_SIZE = 26 + 1
INPUT_SIZE = TEXT_LENGTH * ALPHABET_SIZE

IMG_WIDTH = 100
IMG_HEIGHT = 14
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT

ROOT_DIR = '/home/colin/src/TypographyLearning'


# Header for binary pgm file
HEADER = 'P5\n{} {}\n255\n'.format(IMG_WIDTH, IMG_HEIGHT)