import os
from typing import Optional
import pathlib as plb
import boto3
import sys
import threading
import logging
from botocore.exceptions import ClientError
from argparse import ArgumentParser


logger = logging.getLogger(__name__)
MB = 1024 * 1024
# see https://docs.aws.amazon.com/code-samples/latest/catalog/code-catalog-python-example_code-s3.html


class TransferCallback:
    """
    Handle callbacks from the transfer manager.

    The transfer manager periodically calls the __call__ method throughout
    the upload and download process so that it can take action, such as
    displaying progress to the user and collecting data about the transfer.
    """

    def __init__(self, target_size):
        self._target_size = target_size
        self._total_transferred = 0
        self._lock = threading.Lock()
        self.thread_info = {}

    def __call__(self, bytes_transferred):
        """
        The callback method that is called by the transfer manager.

        Display progress during file transfer and collect per-thread transfer
        data. This method can be called by multiple threads, so shared instance
        data is protected by a thread lock.
        """
        thread = threading.current_thread()
        with self._lock:
            self._total_transferred += bytes_transferred
            if thread.ident not in self.thread_info.keys():
                self.thread_info[thread.ident] = bytes_transferred
            else:
                self.thread_info[thread.ident] += bytes_transferred

            target = self._target_size
            sys.stdout.write(
                f"\r{self._total_transferred / MB} MB of {target} MB transferred "
                f"({(self._total_transferred / MB / target) * 100:.2f}%).")
            sys.stdout.flush()


def download_with_default_configuration(s3_object, download_file_path):
    """
    Download a file from an Amazon S3 bucket to a local folder, using the
    default configuration.
    """
    file_size = s3_object.content_length / MB
    print(f'Downloading ckpt with size {file_size} MB')

    transfer_callback = TransferCallback(file_size)
    s3_object.download_file(download_file_path, Callback=transfer_callback)
    return transfer_callback.thread_info


def list_my_buckets(s3):
    print('Buckets:\n\t', *[b.name for b in s3.buckets.all()], sep="\n\t")


def list_objects(bucket, prefix=None):
    """
    Lists the objects in a bucket, optionally filtered by a prefix.

    Usage is shown in usage_demo at the end of this module.

    :param bucket: The bucket to query.
    :param prefix: When specified, only objects that start with this prefix are listed.
    :return: The list of objects.
    """
    try:
        if not prefix:
            objects = list(bucket.objects.all())
        else:
            objects = list(bucket.objects.filter(Prefix=prefix))
        logger.info("Got objects %s from bucket '%s'",
                    [o.key for o in objects], bucket.name)
    except ClientError:
        logger.exception("Couldn't get objects for bucket '%s'.", bucket.name)
        raise
    else:
        return objects


def fetch_ckpt(hparams, object_key: str = '', bucket_name: Optional[str] = 'my_bucket', download_file_path: str = '', digits=6):
    """
    :param prefix: if only prefix is given, best ckpt is selected
    :param object_key: if object_key is given, particular ckpt is downloaded
    :param bucket_name:
    :param download_file_path:
    :param digits:
    :return:
    """
    prefix = f'checkpoints/{hparams.ckpt_run_name}'

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    list_my_buckets(s3)
    objects = list_objects(bucket, prefix=prefix)
    objects.reverse()  # higher priority to later ckpts
    print(f'Available keys: {[o_.key for o_ in objects]}')

    if not object_key:
        if not hparams.ckpt_select_last:
            # max score
            objects = [o_ for o_ in objects if 'last' not in o_.key]
            score_ = [o_.key[-5 - digits:-5] for o_ in objects]
            if 'loss' in objects[0].key:
                score_of_interest_ = min(score_)
            elif 'dice' in objects[0].key:
                score_of_interest_ = max(score_)
            else:
                ValueError()
            object_selected = s3.Bucket(bucket_name).Object(objects[score_.index(score_of_interest_)].key)  # select best ckpt
        else:
            # max epoch
            selection = [o_ for o_ in objects if 'last' in o_.key][0]
            object_selected = s3.Bucket(bucket_name).Object(selection.key)  # select last ckpt
    else:
        object_selected = s3.Bucket(bucket_name).Object(object_key)
    print(f'Selecting ckpt {object_selected.key}')

    if not download_file_path:
        download_file_path = plb.Path(hparams.ckpt_local_folder) if hparams.ckpt_local_folder else plb.Path(__file__).parent.parent.parent.parent / 'logs' / 'aws'
        download_file_path = str(download_file_path / plb.Path(object_selected.key).parts[-2] / plb.Path(object_selected.key).name)
    os.makedirs(plb.Path(download_file_path).parent, exist_ok=True)
    print(f'Saving ckpt to {download_file_path}')

    if not plb.Path(download_file_path).exists():
        download_with_default_configuration(s3_object=object_selected,
                                            download_file_path=download_file_path)

    return download_file_path


def add_aws_specific_args(parent_parser):
    """
    Specify the hyperparams for this LightningModule
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--ckpt_local_folder', type=str, nargs='?', default='', const='')
    parser.add_argument('--ckpt_select_last', action='store_true')

    return parser


if __name__ == '__main__':
    from collections import namedtuple
    Hparams = namedtuple('hparams', ['ckpt_run_name', 'ckpt_local_folder', 'ckpt_select_last'])
    hparams = Hparams('medical_wip_0716082939', None, True)
    fetch_ckpt(hparams)
