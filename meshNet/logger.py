import os
import sys

import utils
import consts


class StreamLogger(object):
    def __init__(self, log, is_err):
        self.log = log
        if is_err:
            self.stream = sys.stderr
            sys.stderr = self
        else:
            self.stream = sys.stdout
            sys.stdout = self

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Logger(object):
    def __init__(self, sess_info):
        log_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'logs')
        utils.mkdirs(log_dir)

        log_fname = sess_info.title + '_console_log.txt'
        log_full_path = os.path.join(log_dir, log_fname)
        # TODO: Do we need to close the file???
        self.log = open(log_full_path, 'w')

        out_logger = StreamLogger(self.log, False)
        err_logger = StreamLogger(self.log, True)
