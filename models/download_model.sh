FILE=$1
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/models/$FILE.t7
MODEL_FILE=./models/$FILE.t7
wget -N $URL -O $MODEL_FILE
