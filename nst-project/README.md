thor gpu 2
team11
Ft!1806!eR11
ssh team11@thor.cs.uni-saarland.de
printer access code: 8945

Team22, Ft!1806!eR22
Uller login.

# for mounting 
sshfs team11@thor.cs.uni-saarland.de:/localhome/team11/yaseen/fast_neural_style/output-styles/ ./testmount
sshfs team22@uller.cs.uni-saarland.de:/localhome/team22/team11/fast_neural_style/images/ ./testmount

# for unmounting
fusermount -u testmount

scp team11@thor.cs.uni-saarland.de:/localhome/team11/yaseen/fast_neural_style/output-style/pinkgirl_s1mnet.jpg ./


CUDA_VISIBLE_DEVICES=6 python neural_style/neural_style.py --arch tf train --dataset ../val2017 --style-image ./images/style-images/udnie.jpg --save-model-dir ./saved-models --name tf_baseline_udnie --epochs 10 --cuda 1 --checkpoint-model-dir ./saved-models

CUDA_VISIBLE_DEVICES=2 python neural_style/neural_style.py eval --content-image ../val2017/content/amber.jpg --model ./saved-models/third.model --output-image ./output-styles/amber3.jpg --cuda 1

skategirl
CUDA_VISIBLE_DEVICES=2 python neural_style/neural_style.py --arch tfmnet eval --content-image ../val2017/content/000000000785.jpg --model ./saved-models/third.model --output-image ./output-styles/skategirl3.jpg --cuda 1

pinkgirl
CUDA_VISIBLE_DEVICES=2 python neural_style/neural_style.py --arch mnet eval --content-image ../val2017/content/000000009448.jpg --model ./saved-models/epoch_6_mnet_s5.model --output-image ./output-styles/pinkgirltfgarbagee6.jpg --cuda 1

colorful crowd
CUDA_VISIBLE_DEVICES=2 python neural_style/neural_style.py eval --content-image ../val2017/content/000000012670.jpg --model ./saved-models/third.model --output-image ./output-styles/crowdtfnet.jpg --cuda 1

dining table scene
CUDA_VISIBLE_DEVICES=2 python neural_style/neural_style.py --arch tf eval --content-image ../val2017/content/000000018380.jpg --model ./saved-models/epoch_6_tf_s5.model --output-image ./output-styles/diningtablemnete6tf.jpg --cuda 1


