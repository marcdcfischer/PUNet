#!/bin/bash
# Configuration based on selected variants
echo "Using configurations -
training: ${cfg_training},
frozen: ${cfg_frozen},
bias: ${cfg_bias},
labels: ${cfg_labels},
downstream: ${cfg_downstream},
dimensions: ${cfg_dimensions},
architecture: ${cfg_architecture},
prompting: ${cfg_prompting},
adaptation: ${cfg_adaptation},
dataset ${cfg_dataset},
amount: ${cfg_amount}."

# Ablations
# losses / training scheme
# Note: noninstructed means that instructions are not passed to the attention layers. They can still be used for the final similarity comparison
if [ "${cfg_training}" == "meta_self" ]; then
  loss_meta=true
  loss_self=true
  noninstructed=false
elif [ "${cfg_training}" == "meta_self_noninstructed" ]; then
  loss_meta=true
  loss_self=true
  noninstructed=true
elif [ "${cfg_training}" == "meta" ]; then
  loss_meta=true
  loss_self=false
  noninstructed=false
elif [ "${cfg_training}" == "meta_noninstructed" ]; then
  # Only makes sense in combination with fixed layer
  loss_meta=true
  loss_self=false
  noninstructed=true
elif [ "${cfg_training}" == "self" ]; then
  loss_meta=false
  loss_self=true
  noninstructed=false
elif [ "${cfg_training}" == "self_noninstructed" ]; then
  loss_meta=false
  loss_self=true
  noninstructed=true
else
  echo "Invalid training configuration ${cfg_training}."
  exit 1
fi

# (Downstream) frozen scheme
if [ "${cfg_frozen}" == "frozen" ]; then
  frozen=true
elif [ "${cfg_frozen}" == "nonfrozen" ]; then
  frozen=false
else
  echo "Invalid prompting configuration ${cfg_frozen}."
  exit 1
fi

# Bias score schemes
if [ "${cfg_bias}" == "all" ]; then
  bias_instructions=true
  bias_content=true
  bias_vit=false
elif [ "${cfg_bias}" == "image_only" ]; then
  bias_instructions=false
  bias_content=true
  bias_vit=false
elif [ "${cfg_bias}" == "pure" ]; then
  bias_instructions=false
  bias_content=false
  bias_vit=false
elif [ "${cfg_bias}" == "vit" ]; then
  bias_instructions=false
  bias_content=false
  bias_vit=true
else
  echo "Invalid bias configuration ${cfg_bias}."
  exit 1
fi

# Downstream phase
if [ "${cfg_downstream}" == false ]; then
  downstream=false
elif [ "${cfg_downstream}" == true ]; then
  downstream=true
else
  echo "Invalid downstream configuration ${cfg_downstream}."
  exit 1
fi

# 2D / 3D slice configurations
# Note: string with "own" format will be parsed for students: [(1, 2, 3), (4, 5, 6)] = "1,2,3;4,5,6"
if [ "${cfg_dimensions}" == "2d" ]; then
    patch_size_students="224,224,1;160,160,1"
    patch_size_teacher="256 256 1"
    attn_window_size="8 8 1"
elif [ "${cfg_dimensions}" == "3d_flat" ]; then
    patch_size_students="112,112,4;80,80,4"
    patch_size_teacher="128 128 4"
    attn_window_size="6 6 2"
elif [ "${cfg_dimensions}" == "3d_aniso" ]; then
    patch_size_students="56,56,8;40,40,8"
    patch_size_teacher="64 64 8"
    attn_window_size="4 4 4"
elif [ "${cfg_dimensions}" == "3d_large" ]; then
    patch_size_students="112,112,8;80,80,8"
    patch_size_teacher="128 128 8"
    attn_window_size="6 6 4"
else
  echo "Invalid dimensions configuration ${cfg_dimensions}."
  exit 1
fi

# Selectable architectures
if [ "${cfg_architecture}" == "wip" ]; then
  architecture="wip"
elif [ "${cfg_architecture}" == "wip_simple" ]; then
  architecture="wip_simple"
elif [ "${cfg_architecture}" == "unet" ]; then
  architecture="unet"
elif [ "${cfg_architecture}" == "unetr" ]; then
  architecture="unetr"
elif [ "${cfg_architecture}" == "swin_unetr" ]; then
  architecture="swin_unetr"
else
  echo "Invalid architecture configuration ${cfg_architecture}."
  exit 1
fi

# Prompting variants
if [ "${cfg_prompting}" == "full" ]; then
  prompting_variant="full"
elif [ "${cfg_prompting}" == "start" ]; then
  prompting_variant="start"
elif [ "${cfg_prompting}" == "end" ]; then
  prompting_variant="end"
elif [ "${cfg_prompting}" == "encoder" ]; then
  prompting_variant="encoder"
elif [ "${cfg_prompting}" == "decoder" ]; then
  prompting_variant="decoder"
else
  echo "Invalid prompting variant configuration ${cfg_prompting}."
  exit 1
fi

# Body adaptation ablations
if [ "${cfg_adaptation}" == "prompting" ]; then
  adaptation_variant="prompting"
  fixed_output=false
  noninstructed_downstream=false
elif [ "${cfg_adaptation}" == "fixed" ]; then
  adaptation_variant="fixed"
  fixed_output=true
  noninstructed_downstream=true
elif [ "${cfg_adaptation}" == "decoder" ]; then
  adaptation_variant="decoder"
  fixed_output=true
  noninstructed_downstream=true
elif [ "${cfg_adaptation}" == "bias" ]; then
  adaptation_variant="bias"
  fixed_output=true
  noninstructed_downstream=true
elif [ "${cfg_adaptation}" == "adapter" ]; then
  adaptation_variant="adapter"
  fixed_output=true
  noninstructed_downstream=true
elif [ "${cfg_adaptation}" == "bias_prompting" ]; then
  adaptation_variant="bias_prompting"
  fixed_output=false
  noninstructed_downstream=false
else
  echo "Invalid adaptation configuration ${cfg_adaptation}."
  exit 1
fi

# Available datasets
if [ "${cfg_dataset}" == "tcia_btcv" ]; then
  dataset="tcia_btcv"

  # Possible label combinations
  if [ "${cfg_labels}" == "0" ]; then
    # All labels
    label_indices_base="1 2 3 4 5 6 7 8"
    # label_indices_downstream_active=""  # No downstream required
  elif [ "${cfg_labels}" == "1" ]; then
    # Abdominal organs
    label_indices_base="1 2 3 5"
    # label_indices_downstream_active=""  # Any instruction not seen during training is eligible. Set it directly via flag.
  elif [ "${cfg_labels}" == "2" ]; then
    # Digestive system
    label_indices_base="4 6 7 8"
  elif [ "${cfg_labels}" == "-1" ]; then
    # Self-sup only (using fixed instruction - without any seg. loss)
    label_indices_base="1"
    # label_indices_downstream_active=""  # Any instruction not seen during training is eligible. Set it directly via flag.
  else
    echo "Invalid labels configuration ${cfg_labels} for dataset configuration ${cfg_dataset}."
    exit 1
  fi

elif [ "${cfg_dataset}" == "ctorg" ]; then
  dataset="ctorg"

  # Possible label combinations
  if [ "${cfg_labels}" == "0" ]; then
    label_indices_base="1 2 3 4 5"
    # label_indices_downstream_active=""  # No downstream required
  elif [ "${cfg_labels}" == "1" ]; then
    label_indices_base="1 2 4"
    # label_indices_downstream_active=""  # Any instruction not seen during training is eligible. Set it directly via flag.
  elif [ "${cfg_labels}" == "2" ]; then
    label_indices_base="3 5"
    # label_indices_downstream_active=""  # Any instruction not seen during training is eligible. Set it directly via flag.
  elif [ "${cfg_labels}" == "-1" ]; then
    label_indices_base="1"
  else
    echo "Invalid labels configuration ${cfg_labels} for dataset configuration ${cfg_dataset}."
    exit 1
  fi

else
  echo "Invalid dataset configuration ${cfg_dataset}."
  exit 1
fi

# Amount of annotated training data
re='^[0-9]+$'
if [ "${cfg_amount}" == "-1" ]; then
  num_annotated=-1
elif [[ "${cfg_amount}" =~ $re ]] ; then
  num_annotated="${cfg_amount}"
else
  echo "Invalid amount configuration ${cfg_amount}."
  exit 1
fi
