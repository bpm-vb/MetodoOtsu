"""
Benchmark con pyperf su una singola immagine
"""

import pyperf
from skimage import data
from skimage.filters import threshold_multiotsu
from otsu import otsu_threshold_dp

def bench_scikit(loops, image, classes):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        threshold_multiotsu(image, classes)
    return pyperf.perf_counter() - t0

def bench_proposed(loops, image, classes):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        otsu_threshold_dp(image, classes)
    return pyperf.perf_counter() - t0

def add_cmdline_args(cmd, args):
    cmd.extend(("--classes", str(args.classes)))

def run_bench():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument("--classes", type=int, default=3)
    args = runner.parse_args()
    classes = args.classes

    image = data.clock()

    runner.bench_time_func(f"Proposed_Otsu_c{classes}", bench_proposed, image, classes)
    runner.bench_time_func(f"Scikit_Otsu_c{classes}", bench_scikit, image, classes)

if __name__ == "__main__":
    run_bench()