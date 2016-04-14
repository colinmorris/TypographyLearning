#!/bin/bash

for fname in `ls outputs`
do
	base=$(basename "$fname")
	convert -scale 800% outputs/$fname bigoutputs/${base}.png
done
