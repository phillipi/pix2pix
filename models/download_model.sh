FILE=$1
URL=https://people.eecs.berkeley.edu/~junyanz/projects/pix2pix/models/$FILE.t7
MODEL_FILE=./models/$FILE.t7
wget -N $URL -O $MODEL_FILE
