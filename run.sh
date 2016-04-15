# Clear out any previous event data
rm summaries/test/* summaries/train/*
python single_layer.py --train_dir=data/imgs --test_dir=data/rand_imgs --iterations=${1:-2000}
