import os.path as osp
import numpy as np

from mmcv.runner import LoggerHook, master_only


class TensorboardLoggerHook(LoggerHook):
    """
    Hook for starting a tensor-board logger.

    Args:
        log_dir (str or Path): dir to save logger file.
        interval (int): logging interval, default is 10
        ignore_last:
        reset_flag:
        register_logWithIter_keyword:
    """

    def __init__(
            self,
            log_dir=None,
            interval=10,
            ignore_last=True,
            reset_flag=True,
            register_logWithIter_keyword=None
    ):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir
        self.register_logWithIter_keyword = register_logWithIter_keyword

    @master_only
    def before_run(self, runner):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX '
                              'to use TensorboardLoggerHook.')
        else:
            if self.log_dir is None:
                self.log_dir = osp.join(runner.work_dir, 'tf_logs')
            self.writer = SummaryWriter(self.log_dir)

    @master_only
    def single_log(self, tag, record, global_step):
        # self-defined, in format: prefix/suffix_tag
        prefix = tag.split('/')[0]
        suffix_tag = '/'.join(tag.split('/')[1:])
        if prefix == 'image':
            self.writer.add_image(suffix_tag, record, global_step)
            return
        if prefix == 'figure':
            self.writer.add_figure(suffix_tag, record, global_step)
            return
        if prefix == 'histogram':
            self.writer.add_histogram(suffix_tag, record, global_step)
            return
        if prefix == 'scalar':
            self.writer.add_scalar(suffix_tag, record, global_step)
            return

        if isinstance(record, str):
            self.writer.add_text(tag, record, global_step)
            return

        if record.size > 1:
            self.writer.add_image(tag, record, global_step)
        else:
            self.writer.add_scalar(tag, record, global_step)

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = var
            record = runner.log_buffer.output[var]

            global_step = runner.epoch

            # for example, loss will be log as iteration
            if isinstance(self.register_logWithIter_keyword, (tuple, list)):
                for keyword in self.register_logWithIter_keyword:
                    if var.find(keyword) > -1:
                        global_step = runner.iter

            global_step = global_step + 1

            if isinstance(record, (list, tuple)):
                for idx, rec in enumerate(record):
                    tag = var + '/' + '{}'.format(idx)
                    self.single_log(tag, rec, global_step)
            else:
                self.single_log(tag, record, global_step)

    @master_only
    def after_run(self, runner):
        self.writer.close()
