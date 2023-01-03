echo "Using arch: $1 and model $2"
m=`basename $2`
echo $m
for image in ./images/sample/*.jpg; do
img=`basename $image`
echo "Processing: $img"
echo "Output saved as : ${m%.*}_${img}"

op="${m%.*}_${img}"

CUDA_VISIBLE_DEVICES=6 python neural_style/neural_style.py --arch $1 eval --content-image ./images/sample/"$img" --model "$2" --output-image ./images/output-images/"$op" --cuda 1

done
