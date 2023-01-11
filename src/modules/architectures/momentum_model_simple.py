import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from typing import Union, Dict, Optional, Tuple, List
from src.modules.architectures.baseline_unet import MonaiUNet
from src.modules.architectures.baseline_unetr import MonaiUNETR
from src.modules.architectures.baseline_swin_unetr import MonaiSwinUNETR
from src.modules.architectures.swin_unetr_deep import DeepSIAUNetr


class MomentumModelSimple(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf
        self.current_tau_instructions = conf.initial_tau_instructions
        self.current_tau_body = conf.initial_tau_body

        # Architecture
        if self.conf.architecture.lower() == 'wip_simple':
            self.architecture = DeepSIAUNetr
        elif self.conf.architecture.lower() == 'unet':
            self.architecture = MonaiUNet
        elif self.conf.architecture.lower() == 'unetr':
            self.architecture = MonaiUNETR
        elif self.conf.architecture.lower() == 'swin_unetr':
            self.architecture = MonaiSwinUNETR
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
                x_teacher: Optional[torch.Tensor] = None):

        dict_out_students = [self.network_student(x_) for x_ in x]
        dict_out_teacher = None
        if x_teacher is not None:
            dict_out_teacher = self.network_teacher(x_teacher)

        return dict_out_students, dict_out_teacher

    def update_teacher(self):
        # Apply momentum weight update
        # Note: batch norms are in general left untouched (i.e. are updated separately)
        for (name, param_student_), (_, param_teacher_) in zip(
            self.network_student.named_parameters(),
            self.network_teacher.named_parameters(),
        ):
            param_teacher_.data = self.current_tau_body * param_teacher_.data + (1 - self.current_tau_body) * param_student_.data

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        hparams_tmp = parser.parse_known_args()[0]
        if hparams_tmp.architecture.lower() == 'wip_simple':
            parser = DeepSIAUNetr.add_model_specific_args(parser)
        elif hparams_tmp.architecture.lower() == 'unet':
            parser = MonaiUNet.add_model_specific_args(parser)
        elif hparams_tmp.architecture.lower() == 'unetr':
            parser = MonaiUNETR.add_model_specific_args(parser)
        elif hparams_tmp.architecture.lower() == 'swin_unetr':
            parser = MonaiSwinUNETR.add_model_specific_args(parser)
        else:
            raise NotImplementedError(f'The selected architecture {hparams_tmp.architecture} is not available.')

        # Momentum
        parser.add_argument('--initial_tau_instructions', default=0.99, type=float)
        parser.add_argument('--initial_tau_body', default=0.99, type=float)

        return parser
