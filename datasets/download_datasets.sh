FILE=$1
URL=https://people.eecs.berkeley.edu/~isola/pix2pix/$FILE.tar
TAR_FILE=./datasets/$FILE.tar
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -xvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
