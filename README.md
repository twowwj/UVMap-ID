# UVMap-ID

## DATA Preparation
# Download the pre-trained checkpoints from Google drive link as follows:
gdown https://drive.google.com/uc?id=1Bw0WRuRtf_czwtL6ZtB5MlI2F45StSo0

## instruction
bash scripts/train_dreambooth_control.sh  (dreambooth + controlnet)

## re-train the projection model from InStantID
bash scripts/train_dreambooth_control_faceid.sh (dreambooth + pretrained controlnet(ours) + projection module(InStantID))

## load the pretrained ip-adapter
bash scripts/train_dreambooth_control_faceid_ipa.sh (dreambooth + pretrained controlnet(ours) + faceipa)