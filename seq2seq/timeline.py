"""Module with tools for timeline tracking."""

import os

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.training import basic_session_run_hooks


def save_timeline(path, run_metadata):
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(path, 'w') as f:
        f.write(chrome_trace)


class TimelineHook(tf.train.SessionRunHook):

    def __init__(self, timeline_dir, every_n_iter=None, every_n_secs=None):
        if (every_n_iter is None and every_n_secs is None) or (
            every_n_iter is not None and every_n_secs is not None):
            raise ValueError(
                "Either every_n_iter or every_n_secs should be used.")
        self._timeline_dir = timeline_dir
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_secs=every_n_secs, every_steps=every_n_iter)
        self._iter_count = 0

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            return tf.train.SessionRunArgs([], options=options)
        return None

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            self._timer.update_last_triggered_step(self._iter_count)
            save_timeline(os.path.join(
                self._timeline_dir, "timeline-%d.json" % self._iter_count),
                run_values.run_metadata)
        self._iter_count += 1

