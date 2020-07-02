#!/usr/bin/env python

import argparse
import os
from . import jobqueue


def run_single():
    parser = argparse.ArgumentParser(description="Submit a single job to queue")

    parser.add_argument('input_file', nargs='+')

    jobqueue.addArguments(parser)

    args = parser.parse_args()

    ini = [ini.replace('.ini', '').replace('.yaml', '') for ini in args.input_file]

    jobqueue.submitJob(os.path.basename(ini[0]), ini, msg=True, **args.__dict__)


if __name__ == "__main__":
    run_single()
