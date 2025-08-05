# # PARAMS
MODEL_REPO="Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Qwen2.5-Math-1.5B-Instruct"
# MODEL_REPO="Qwen/Qwen2.5-1.5B"
# MODEL_NAME="Qwen2.5-1.5B"
N=8
OUTPUT_DIR=results/steering/$(basename "$MODEL_REPO")$
START=0
END=2000
STEERN=500

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/math
mkdir -p $OUTPUT_DIR/gsm8k/
mkdir -p $OUTPUT_DIR/aime24
mkdir -p $OUTPUT_DIR/olympiadbench

# Run once with a large N for AIME24
TEMPERATURE=0.8
for LAYER in 20 21 22 23; do
  for CALPHA_K in 1; do
  echo "Running Steering algorithm1 VER 3 on AIME24 dataset..."
  python experiments/temperature_tuning/run_steering_algor1_ver3-tune-alphak_after.py \
    --model_repo "$MODEL_REPO" \
    --data_path "data/aime24/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --recalc_steer_after_n_tokens $STEERN \
    --temperature $TEMPERATURE \
    --calpha_k $CALPHA_K \
    --save_all_candidates \
    --steer_at_layer $LAYER \
    --output_dir $OUTPUT_DIR/aime24/temperature-sampling-fixed-layer \
    --run_name_before "steer-algor1-ver3_n8_aime24_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-layer${LAYER}-before" \
    --run_name_after "steer-algor1-ver3_n8_aime24_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-layer${LAYER}-after"
  done
done
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/$MODEL_NAME$/aime24/temperature-sampling-fixed-layer   --plot --dataset "aime24"   --output_dir /home/ly/DataDistillation/results/steering/$MODEL_NAME$/aime24/temperature-sampling-fixed-layer


# Run once with a large N for OLYMPIADBENCH
# for TEMPERATURE in 0.4; do
#   for CALPHA_K in 10; do
#   echo "Running Steering algorithm1 VER 3 on OlympiadBench dataset..."
#   python experiments/temperature_tuning/run_steering_algor1_ver3-tune-alphak_after.py \
#     --model_repo "$MODEL_REPO" \
#     --data_path "data/olympiadbench/test.jsonl" \
#     --input_start $START \
#     --input_end $END \
#     --number_candidate $N \
#     --recalc_steer_after_n_tokens $STEERN \
#     --temperature $TEMPERATURE \
#     --calpha_k $CALPHA_K \
#     --save_all_candidates \
#     --output_dir $OUTPUT_DIR/olympiadbench/temperature-sampling-fixed-15 \
#     --run_name_before "steer-algor1-ver3_n8_olympiadbench_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-before" \
#     --run_name_after "steer-algor1-ver3_n8_olympiadbench_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-after"
#     python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/$MODEL_NAME$/olympiadbench/temperature-sampling-fixed-15   --plot --dataset "olympiadbench"   --output_dir /home/ly/DataDistillation/results/steering/$MODEL_NAME$/olympiadbench/temperature-sampling-fixed-15
#   done
# done

# for TEMPERATURE in 0.6 0.8 1.0; do
#   for CALPHA_K in 1 10; do
#   echo "Running Steering algorithm1 VER 3 on MATH500 dataset..."
#   python experiments/temperature_tuning/run_steering_algor1_ver3-tune-alphak_after.py \
#     --model_repo "$MODEL_REPO" \
#     --data_path "data/math/test.jsonl" \
#     --input_start $START \
#     --input_end $END \
#     --number_candidate $N \
#     --recalc_steer_after_n_tokens $STEERN \
#     --temperature $TEMPERATURE \
#     --calpha_k $CALPHA_K \
#     --save_all_candidates \
#     --output_dir $OUTPUT_DIR/math/temperature-sampling-fixed-15 \
#     --run_name_before "steer-algor1-ver3_n$N-math_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-before" \
#     --run_name_after "steer-algor1-ver3_n$N-math_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-after"
#   done
#   python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/$MODEL_NAME$/math/temperature-sampling-fixed-15  --plot --dataset "math"   --output_dir /home/ly/DataDistillation/results/steering/$MODEL_NAME$/math/temperature-sampling-fixed-15
# done



