#!/bin/bash

# Check amount of arguments
if [[ $# -lt 14 ]] ; then
    echo 'A wrong amount of arguments (< 14) has been provided.'
    exit 1
fi

# load env variables
username="${14}"
CLUSTER="my_cluster"  # EDIT ME
PYTHON="/my/python/versions/3.9.8/bin/python"  # EDIT ME
CODE="/my/code/"  # EDIT ME

export PYTHONPATH="${CODE}"  # EDIT ME

# Parse arguments as GPU list
export CUDA_VISIBLE_DEVICES="$1"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
gpus="--gpus 1"

# set main params
script="main.py"
dir_images="--dir_images /my/data/"  # EDIT ME
dir_masks="--dir_masks /my/data/"  # EDIT ME

# default parameters
max_epochs=400
max_epochs_downstream=100
batch_size=16
learning_rate=1e-4
learning_rate_downstream=5e-4
learning_rate_instructions=1e-3
learning_rate_instructions_downstream=5e-4
loss_weight_segmentation=1e-0
loss_weight_sim_protos=1e-2
tokens_per_instruction=16
num_workers=16

# configurations
cfg_training="$2"
cfg_frozen="$3"
cfg_bias="$4"
cfg_labels="$5"
cfg_downstream="$6"
cfg_dimensions="$7"
cfg_architecture="$8"
cfg_prompting="$9"
cfg_adaptation="${10}"
cfg_dataset="${11}"
cfg_amount="${12}"
source ../common/cfgs.sh

# fetch checkpoint (if available)
ckpt=""  # "--ckpt "
if [ "${13}" == "none" ]; then
  echo "Using no ckpt."
else
  ckpt="--ckpt_run_name ${13}"
  echo "Using ckpt ${13}."
fi

# ablations
ablations="${*:15}"  # Anything passed (directly) as flag during shell script call.
ablations="${ablations}"
if [ "$loss_meta" == false ]; then loss_weight_segmentation=0.; fi;
if [ "$loss_self" == false ]; then loss_weight_sim_protos=0.; fi;
if [ "$downstream" == true ]; then ablations="${ablations} --downstream --no_overwrite --cold_start"; fi;
if [ "${architecture}" == "wip" ] || [ "${architecture}" == "wip_simple" ]; then
  if [ "$noninstructed" == true ]; then ablations="${ablations} --noninstructed_attention"; fi;
  if [ "$noninstructed_downstream" == true ]; then ablations="${ablations} --noninstructed_attention_downstream"; fi;
  if [ "$frozen" == true ]; then ablations="${ablations} --selective_freezing"; fi;
  if [ "$bias_instructions" == false ]; then ablations="${ablations} --no_bias_instructions"; fi;
  if [ "$bias_content" == false ]; then ablations="${ablations} --no_bias_content"; fi;
  if [ "$bias_vit" == true ]; then ablations="${ablations} --bias_vit"; fi;
  if [ "$fixed_output" == true ]; then ablations="${ablations} --fixed_output"; fi;
fi;

# parameters (potentially adjusted by above ablation statements)
parameters="--batch_size ${batch_size} --learning_rate ${learning_rate} --learning_rate_downstream ${learning_rate_downstream} --learning_rate_instructions ${learning_rate_instructions} --learning_rate_instructions_downstream ${learning_rate_instructions_downstream}"
parameters="${parameters} --architecture ${architecture}"
parameters="${parameters} --loss_weight_segmentation ${loss_weight_segmentation} --loss_weight_sim_protos ${loss_weight_sim_protos}"
parameters="${parameters} --dataset ${dataset} --num_workers ${num_workers}"
parameters="${parameters} --num_annotated ${num_annotated}"
if [ "$downstream" == true ]; then parameters="${parameters} --max_epochs ${max_epochs_downstream}"; else parameters="${parameters} --max_epochs ${max_epochs}"; fi;
if [ "${architecture}" == "wip" ] || [ "${architecture}" == "wip_simple" ]; then
  parameters="${parameters} --label_indices_base ${label_indices_base} --tokens_per_instruction ${tokens_per_instruction}"
  parameters="${parameters} --patch_size_students ${patch_size_students} --patch_size_teacher ${patch_size_teacher} --attn_window_size ${attn_window_size}"
  parameters="${parameters} --prompting_variant ${prompting_variant} --adaptation_variant ${adaptation_variant}"
fi;

# tags
tags="--tags"
# tags from configuration
tags="${tags} dim_${cfg_dimensions} nn_${architecture} data_${dataset} user_${username} cl_${CLUSTER}"
tags="${tags} lr_${learning_rate} lrds_${learning_rate_downstream} lri_${learning_rate_instructions} lrids_${learning_rate_instructions_downstream} bs_${batch_size} na_${num_annotated}"
if [ "$loss_meta" == true ]; then tags="${tags} loss_meta"; fi;
if [ "$loss_self" == true ]; then tags="${tags} loss_self"; fi;
if [ "${13}" != "none" ]; then tags="${tags} ckpt_${13:17}"; fi;
if [ "$downstream" == true ]; then tags="${tags} downstream"; fi;
if [ "${architecture}" == "wip" ] || [ "${architecture}" == "wip_simple" ]; then
  tags="${tags} tk_${tokens_per_instruction} pv_${prompting_variant} av_${adaptation_variant}"
  if [ "$noninstructed" == true ]; then tags="${tags} noninstructed"; fi;
  if [ "$noninstructed_downstream" == true ]; then tags="${tags} noninstructed_ds"; fi;
  if [ "$fixed_output" == true ]; then tags="${tags} fixed_output"; fi;
  if [ "$frozen" == true ]; then tags="${tags} frozen"; fi;
  if [ "$bias_instructions" == true ]; then tags="${tags} bias_instructions"; fi;
  if [ "$bias_content" == true ]; then tags="${tags} bias_content"; fi;
  if [ "$bias_vit" == true ]; then tags="${tags} bias_vit"; fi;
fi;

# Run python code
timestamp=$(date +%Y%m%d_%H%M%S)
std_out="std_${timestamp}.out"
std_err="std_${timestamp}.err"
echo "Using python: ${PYTHON}"
echo "Executing cmd: $PYTHON ${CODE}/src/${script} ${dir_images} ${dir_masks} ${gpus} ${parameters} ${ablations} ${tags} ${ckpt} > ${std_out} 2> ${std_err}"
$PYTHON ${CODE}/src/${script} ${dir_images} ${dir_masks} ${gpus} ${parameters} ${ablations} ${tags} ${ckpt} > ${std_out} 2> ${std_err}
