import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from typing import Union, Dict, Optional, Tuple, List
# from src.modules.architectures.interpreter import Interpreter as Architecture
from src.modules.architectures.swin_unetr_deep import DeepSIAUNetr
import warnings


class MomentumModel(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf
        self.current_tau_instructions = conf.initial_tau_instructions
        self.current_tau_body = conf.initial_tau_body

        # Architecture
        if self.conf.architecture_wip.lower() == 'shallow':
            raise ValueError('Model has been removed, since its implementation was outdated. ')
        elif self.conf.architecture_wip.lower() == 'deep':
            self.architecture = DeepSIAUNetr
        else:
            raise NotImplementedError(f'The selected architecture {self.conf.architecture} is not available.')

        # Base architecture (x2). Anything in there will be replicated 2x.
        self.network_student = self.architecture(conf=self.conf)
        self.network_teacher = self.architecture(conf=self.conf)

        # Overwrite teacher initialization with teachers one and disable gradients
        for (name, param_student_), (_, param_teacher_) in zip(
            self.network_student.named_parameters(),
            self.network_teacher.named_parameters(),
        ):
            param_teacher_.data.copy_(param_student_.data)  # initialize teacher with identical data as student
            param_teacher_.requires_grad = False  # Do not update by gradient

    def forward(self,
                x: List[torch.Tensor],
                x_teacher: Optional[torch.Tensor] = None,
                label_indices: Optional[torch.Tensor] = None,
                pseudo_indices_subject: Optional[torch.Tensor] = None,
                pseudo_indices_label: Optional[torch.Tensor] = None,
                mode_label: str = 'label',
                mode_loss: str = 'both'):

        dict_out_students = [self.network_student(x_, label_indices=label_indices, pseudo_indices_subject=pseudo_indices_subject, pseudo_indices_label=pseudo_indices_label, mode_label=mode_label, mode_loss=mode_loss) for x_ in x]
        dict_out_teacher = None
        if x_teacher is not None:
            dict_out_teacher = self.network_teacher(x_teacher, label_indices=label_indices, pseudo_indices_subject=pseudo_indices_subject, pseudo_indices_label=pseudo_indices_label, mode_label=mode_label, mode_loss=mode_loss)

        return dict_out_students, dict_out_teacher

    def update_teacher(self):
        # Apply momentum weight update
        # Note: batch norms are in general left untouched (i.e. are updated separately)
        for (name, param_student_), (_, param_teacher_) in zip(
            self.network_student.named_parameters(),
            self.network_teacher.named_parameters(),
        ):
            if 'instruction_pool' in name:
                param_teacher_.data = self.current_tau_instructions * param_teacher_.data + (1 - self.current_tau_instructions) * param_student_.data
            else:
                param_teacher_.data = self.current_tau_body * param_teacher_.data + (1 - self.current_tau_body) * param_student_.data

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Architecture
        hparams_tmp = parser.parse_known_args()[0]
        if hparams_tmp.architecture_wip.lower() == 'shallow':
            raise ValueError('Model has been removed, since its implementation was outdated. ')
        elif hparams_tmp.architecture_wip.lower() == 'deep':
            parser = DeepSIAUNetr.add_model_specific_args(parser)
        else:
            raise NotImplementedError(f'The selected architecture {hparams_tmp.architecture_wip} is not available.')

        # Momentum
        parser.add_argument('--initial_tau_instructions', default=0.99, type=float)
        parser.add_argument('--initial_tau_body', default=0.99, type=float)

        return parser
