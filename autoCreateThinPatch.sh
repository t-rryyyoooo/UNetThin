#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createThinPatch.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="createThinPatch.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".label_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly IMAGE_PATCH_WIDTH=$(cat ${JSON_FILE} | jq -r ".image_patch_width")
readonly LABEL_PATCH_WIDTH=$(cat ${JSON_FILE} | jq -r ".label_patch_width")
readonly PLANE_SIZE=$(cat ${JSON_FILE} | jq -r ".plane_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")


for number in ${NUM_ARRAY[@]}
do
    image_path="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
    label_path="${DATA_DIRECTORY}/case_${number}/${LABEL_NAME}"
    save_path="${SAVE_DIRECTORY}/case_${number}"
    echo "image_path:${image_path}"
    echo "label_path:${label_path}"
    echo "SAVE_PATH:${save_path}"
    echo "IMAGE_PATCH_WIDTH:${IMAGE_PATCH_WIDTH}"
    echo "LABEL_PATCH_WIDTH:${LABEL_PATCH_WIDTH}"
    echo "PLANE_SIZE:${PLANE_SIZE}"
    echo "OVERLAP:${OVERLAP}"

    if [ $MASK_NAME = "No" ]; then
     mask=""

    else
     mask_path="${DATA_DIRECTORY}/case_${number}/${MASK_NAME}"
     echo "MASK_PATH:${mask_path}"
     mask="--mask_path ${mask_path}"

    fi

    python3 createThinPatch.py ${image_path} ${label_path} ${save_path} --image_patch_width ${IMAGE_PATCH_WIDTH} --label_patch_width ${LABEL_PATCH_WIDTH} --plane_size ${PLANE_SIZE} --overlap ${OVERLAP} ${mask}

# Judge if it works.
    if [ $? -eq 0 ]; then
     echo "Done."

    else
     echo "Fail"

    fi
done
