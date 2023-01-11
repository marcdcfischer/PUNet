#!/bin/bash
# call via nohup ./train_p1.sh 0 &

cd "$(dirname ${0})" || exit

gpu=$1
# See common/cfgs.sh for all options
cfg_training=("meta_self")
cfg_frozen=("nonfrozen")
cfg_bias=("all")
cfg_labels=("0")
cfg_downstream=(false)
cfg_dimensions=("2d")
cfg_architecture=("wip")
cfg_prompting=("full")
cfg_adaptation=("prompting")
cfg_dataset=("tcia_btcv")  # tcia_btcv or ctorg
cfg_amount=("-1")  # use half the amount of annotated for tcia_btcv since those are two datasets
ckpts=("none")  # "none" or (w&b) run name
username="github-user"
misc="${*:2}"  # fetch remaining parameters

# loop through runs
timestamp=$(date +%Y%m%d_%H%M%S)
multiple_out="multiple_${timestamp}.out"
multiple_err="multiple_${timestamp}.err"
for idx_ in {0..0}; do
  sleep 2
  echo "bash ./train.sh ${gpu} ${cfg_training[idx_]} ${cfg_frozen[idx_]} ${cfg_bias[idx_]} ${cfg_labels[idx_]} ${cfg_downstream[idx_]} ${cfg_dimensions[idx_]} ${cfg_architecture[idx_]} ${cfg_prompting[idx_]} ${cfg_adaptation[idx_]} ${cfg_dataset[idx_]} ${cfg_amount[idx_]} ${ckpts[idx_]} ${username} ${misc} >> ${multiple_out} 2>> ${multiple_err} &" >> ${multiple_out} 2>> ${multiple_err}
  bash ./train.sh ${gpu} ${cfg_training[idx_]} ${cfg_frozen[idx_]} ${cfg_bias[idx_]} ${cfg_labels[idx_]} ${cfg_downstream[idx_]} ${cfg_dimensions[idx_]} ${cfg_architecture[idx_]} ${cfg_prompting[idx_]} ${cfg_adaptation[idx_]} ${cfg_dataset[idx_]} ${cfg_amount[idx_]} ${ckpts[idx_]} ${username} ${misc} >> ${multiple_out} 2>> ${multiple_err} &
  wait
done
