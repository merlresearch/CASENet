# Copyright (C) 2017, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument("solver_prototxt_file", type=str, help="path to the solver prototxt file")
parser.add_argument(
    "-c",
    "--pycaffe_folder",
    type=str,
    default="../../code/python",
    help="pycaffe folder that contains the caffe/_caffe.so file",
)
parser.add_argument(
    "-m", "--init_model", type=str, default="./model/init_res_coco.caffemodel", help="path to the initial caffemodel"
)
parser.add_argument("-g", "--gpu", type=int, default=0, help="use which gpu device (default=0)")
args = parser.parse_args(sys.argv[1:])

assert os.path.exists(args.solver_prototxt_file)
assert os.path.exists(args.init_model)

if os.path.exists(os.path.join(args.pycaffe_folder, "caffe/_caffe.so")):
    sys.path.insert(0, args.pycaffe_folder)
import caffe

caffe.set_mode_gpu()
caffe.set_device(args.gpu)

solver = caffe.SGDSolver(args.solver_prototxt_file)
solver.net.copy_from(args.init_model)

solver.solve()
