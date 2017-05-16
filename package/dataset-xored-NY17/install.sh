#! /bin/bash

# PACKAGE_DIR
# INSTALL_DIR

# KITTI_URL

echo "Downloading images..."

mkdir $IMAGE_DIR
cd $IMAGE_DIR

echo "Downloading files..."
smbclient '\\turbine\share' -N -c "prompt OFF;recurse ON;cd \"Photo/2016-12-23 - Xored New Year Party '17\";lcd \"$PWD\";mget *"

echo "Padding filenames with zeroes..."
for a in [0-9]*.jpg; do mv $a `printf %06d.%s ${a%.*} ${a##*.}`; done

echo "Filtering out vertical images..."
for a in [0-9]*.jpg; do 
  w=`convert $a -print "%w" /dev/null`
  h=`convert $a -print "%h" /dev/null`
  if [ $h -gt $w ]; then rm $a; fi
done

exit 0
