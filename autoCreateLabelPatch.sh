#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createLabelPatch.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="createLabelPatch.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly LABEL_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".label_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".label_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly IS_LABEL=$(cat ${JSON_FILE} | jq -r ".is_label")
readonly SAVE_NAME=$(cat ${JSON_FILE} | jq -r ".save_name")
readonly PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".patch_size")
readonly PLANE_SIZE=$(cat ${JSON_FILE} | jq -r ".plane_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly NUM_REP=$(cat ${JSON_FILE} | jq -r ".num_rep")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")


for number in ${NUM_ARRAY[@]}
do
    label_path="${LABEL_DIRECTORY}/case_${number}/${LABEL_NAME}"
    save_path="${SAVE_DIRECTORY}/case_${number}"
    echo "LABEL_PATH:${label_path}"
    echo "IS_LABEL:${IS_LABEL}"
    echo "SAVE_PATH:${save_path}"
    echo "PATCH_SIZE:${PATCH_SIZE}"
    echo "PLANE_SIZE:${PLANE_SIZE}"
    echo "OVERLAP:${OVERLAP}"
    echo "NUM_REP:${NUM_REP}"

    if [ $MASK_NAME = "No" ]; then
     mask=""

    else
     mask_path="${LABEL_DIRECTORY}/case_${number}/${MASK_NAME}"
     echo "MASK_PATH:${mask_path}"
     mask="--mask_path ${mask_path}"

    fi

    if "${IS_LABEL}"; then
     echo "Is_label!"
     is_label="--is_label"
     
    else
     is_label=""

    fi

    python3 createLabelPatch.py ${label_path} ${save_path} --patch_size ${PATCH_SIZE} --plane_size ${PLANE_SIZE} --overlap ${OVERLAP} --save_image --num_rep ${NUM_REP} ${is_label} ${mask}

# Judge if it works.
    if [ $? -eq 0 ]; then
     echo "Done."

    else
     echo "Fail"

    fi
done
