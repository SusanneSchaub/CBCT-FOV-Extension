#!/bin/bash
# Run this from the ddpm_model folder

USAGE_TEXT="Usage: ./run train|sample|resume [additional_arguments]"
if [[ -z "$1" ]]; then
    echo "$USAGE_TEXT"
    exit 1
elif [[ "$1" == "sample" ]]; then
    echo "Input: sample. OK."
elif [[ "$1" == "train" ]]; then
    echo "Input: train. OK."
elif [[ "$1" == "resume" ]]; then
    echo "Input: resume. OK."
elif [[ "$1" == "watch" ]]; then
    echo "Input: watch. OK."
else
    echo "$USAGE_TEXT"
    echo "Invalid Input. Leaving..."
    exit 1
fi

# GENERAL VARIABLES
LOG_DIR=""
SAMPLE_DIR="samples"
MODEL_DIR="models"

# MODEL FLAGS
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 14 --random_masks False --augment_data True --log_dir $LOG_DIR"
SAMPLE_FLAGS="--random_masks False --log_dir $SAMPLE_DIR"

PROJECT_ROOT=~/inr-diffusion/src

TRAIN_SCRIPT="$PROJECT_ROOT/cbct_artifact_reduction/scripts/train_test.py"
SAMPLE_SCRIPT="$PROJECT_ROOT/cbct_artifact_reduction/scripts/sample_test_series.py"

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH


MODEL_PATH1="xxx.pt"


# Catch all extra arguments after train|sample|resume|watch
EXTRA_ARGS="${*:2}"

# PRINTING VARIABLES
echo "Logging to: $LOG_DIR"

echo "--------------------"
echo "MODEL_FLAGS:"
echo "$MODEL_FLAGS"
echo "--------------------"
echo "DIFFUSION_FLAGS:"
echo "$DIFFUSION_FLAGS"
echo "--------------------"
echo "TRAIN_FLAGS:"
echo "$TRAIN_FLAGS"
echo "--------------------"
echo "SAMPLE_FLAGS:"
echo "$SAMPLE_FLAGS"
echo "--------------------"
echo "EXTRA_ARGS:"
echo "$EXTRA_ARGS"
echo "--------------------"

if [[ "$1" == "train" ]]; then
    echo "Starting training"
    echo "Running $TRAIN_SCRIPT"
    python $TRAIN_SCRIPT $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $EXTRA_ARGS
elif [[ "$1" == "resume" ]]; then
    echo "Resume training"
    echo "Using model: $MODEL_PATH"
    echo "Running $TRAIN_SCRIPT"
    python $TRAIN_SCRIPT --resume_checkpoint $MODEL_PATH $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $EXTRA_ARGS
elif [[ "$1" == "sample" ]]; then
    echo "Starting sampling"
    echo "Running $SAMPLE_SCRIPT"
    python $SAMPLE_SCRIPT --model_path1 $MODEL_PATH1 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $EXTRA_ARGS
elif [[ "$1" == "watch" ]]; then
    echo "Starting continuously sampling"
    echo "Running $CONTINUOUSLY_SAMPLE_SCRIPT"
    python $CONTINUOUSLY_SAMPLE_SCRIPT $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $EXTRA_ARGS
fi
