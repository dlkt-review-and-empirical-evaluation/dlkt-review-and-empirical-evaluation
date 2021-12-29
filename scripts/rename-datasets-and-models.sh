#!/bin/sh

filename="$1"

# Datasets
sed -i -E 's/(assist09up)|(assist09-updated)/ASSISTments 2009 Updated/' "$filename"
sed -i -E 's/(assist15)|(2015_100_skill_builders_main_problems)/ASSISTments 2015/' "$filename"
sed -i -E 's/(assist17)|(assist17-challenge)/ASSISTments 2017/' "$filename"
sed -i -E 's/(intro-prog)|(Intro-Prog)/IntroProg/' "$filename"
sed -i -E 's/(stat)|(statics)/Statics/' "$filename"
sed -i -E 's/(synth-k)|(synthetic-5-k)/Synthetic-K/' "$filename"
sed -i -E 's/Assist/ASSIST/' "$filename"
sed -i -E 's/-([0-9]+)-/ \1 /' "$filename"
sed -i -E 's/-([0-9]+),/ \1,/' "$filename"

# Baseline Models
sed -i 's/next as previous/NaP/' "$filename"
sed -i 's/mean/Mean/' "$filename"
sed -i 's/lrbest/GLR/' "$filename"

# DLKT Models
sed -i -E 's/(lstm)/\U\1/' "$filename"
sed -i -E 's/(dkt)/\U\1/' "$filename"
sed -i -E 's/(dkvmn)/\U\1/' "$filename"
sed -i -E 's/(sakt)/\U\1/' "$filename"
sed -i 's/vanilla/Vanilla/' "$filename"
sed -i 's/paper/Paper/' "$filename"
sed -i 's/-s[+],/-S+,/' "$filename"
