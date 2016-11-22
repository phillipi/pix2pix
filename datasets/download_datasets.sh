FILE=$1
URL=https://people.eecs.berkeley.edu/~isola/pix2pix/$FILE.tar
TAR_FILE=./datasets/$FILE.tar
TARGET_DIR=./datasets/$FILE/
wget $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -xvf $TAR_FILE -C $TARGET_DIR
rm $TAR_FILE
