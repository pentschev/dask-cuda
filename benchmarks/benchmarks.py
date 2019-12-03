# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from collections import defaultdict
import argparse
import asyncio
from time import perf_counter as clock
import numpy as np
import cupy
from pprint import pprint

import dask.array as da
from dask.distributed import Client, wait
from dask.utils import format_time, format_bytes, parse_bytes
from dask_cuda import LocalCUDACluster, DGX


async def run(args):

    # Set up workers on the local machine
    async with DGX(
        protocol=args["protocol"],
        n_workers=len(args["devs"].split(",")),
        CUDA_VISIBLE_DEVICES=args["devs"],
        enable_tcp_over_ucx=args["enable_tcp_over_ucx"],
        enable_nvlink=args["enable_nvlink"],
        enable_infiniband=args["enable_infiniband"],
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:

            # Create a simple random array
            rs = da.random.RandomState(RandomState=cupy.random.RandomState)
            x = rs.random((args["size"], args["size"]), chunks=args["chunk_size"]).persist()
            await wait(x)
            t1 = clock()
            await client.compute((x + x.T).sum())
            took = clock() - t1

            # Collect, aggregate, and print peer-to-peer bandwidths
            incoming_logs = await client.run(
                lambda dask_worker: dask_worker.incoming_transfer_log
            )
            bandwidths = defaultdict(list)
            total_nbytes = defaultdict(list)
            for k, L in incoming_logs.items():
                for d in L:
                    if d["total"] >= args["ignore_size"]:
                        bandwidths[k, d["who"]].append(d["bandwidth"])
                        total_nbytes[k, d["who"]].append(d["total"])
            bandwidths = {
                (
                    cluster.scheduler.workers[w1].name,
                    cluster.scheduler.workers[w2].name,
                ): [
                    "%s/s" % format_bytes(x) for x in np.quantile(v, [0.25, 0.50, 0.75])
                ]
                for (w1, w2), v in bandwidths.items()
            }
            total_nbytes = {
                (
                    cluster.scheduler.workers[w1].name,
                    cluster.scheduler.workers[w2].name,
                ): format_bytes(sum(nb))
                for (w1, w2), nb in total_nbytes.items()
            }

            print("Roundtrip benchmark")
            print("--------------------------")
            print(f"Size        | {args['size']}*{args['size']}")
            print(f"Chunk-size  | {args['chunk_size']}")
            print(f"Ignore-size | {format_bytes(args['ignore_size'])}")
            print(f"Protocol    | {args['protocol']}")
            print(f"Device(s)   | {args['devs']}")
            print(f"npartitions | {x.npartitions}")
            print("==========================")
            print(f"Total time  | {format_time(took)}")
            print("==========================")
            print("(w1,w2)     | 25% 50% 75% (total nbytes)")
            print("--------------------------")
            for (d1, d2), bw in sorted(bandwidths.items()):
                print(
                    "(%02d,%02d)     | %s %s %s (%s)"
                    % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)])
                )


def time_local_cuda(protocol_options, size):

    #args = {"protocol": "tcp", "devs": "0,1", "size": 10000, "chunk_size": 134217728,
    #        "ignore_size": 1048576}
    #args = {"protocol": "ucx", "devs": "0,1", "size": 10000, "chunk_size": 134217728,
    #        "ignore_size": 1048576}
    #args = {"protocol": "tcp", "devs": "0,1", "size": 10000, "chunk_size": "128 MiB",
    #        "ignore_size": "1 MiB"}
    protocol = protocol_options[0]
    enable_tcp_over_ucx = protocol_options[1]
    enable_nvlink = protocol_options[2]
    enable_infiniband = protocol_options[3]
    args = {"protocol": protocol, "devs": "0,1", "size": size, "chunk_size": 134217728,
            "ignore_size": 1048576, "enable_tcp_over_ucx": enable_tcp_over_ucx,
            "enable_nvlink": enable_nvlink, "enable_infiniband": enable_infiniband}
    asyncio.get_event_loop().run_until_complete(run(args))

time_local_cuda.params = ([("tcp", False, False, False), ("ucx", True, False, False),
                           ("ucx", True, True, False)],
                          [10000, 40000])
time_local_cuda.params_names = ["protocol", "size"]

#class TimeSuite:
#    """
#    An example benchmark that times the performance of various kinds
#    of iterating over dictionaries in Python.
#    """
#    params = (["tcp", "ucx"], [10000, 40000])
#
#    def setup(self, n):
#        self.obj = range(n)
#
#    def teardown(self, n):
#        del self.obj
#
#    def time_local_cuda(self, n):
#
#        #args = {"protocol": "tcp", "devs": "0,1", "size": 10000, "chunk_size": 134217728,
#        #        "ignore_size": 1048576}
#        #args = {"protocol": "ucx", "devs": "0,1", "size": 10000, "chunk_size": 134217728,
#        #        "ignore_size": 1048576}
#        #args = {"protocol": "tcp", "devs": "0,1", "size": 10000, "chunk_size": "128 MiB",
#        #        "ignore_size": "1 MiB"}
#        for i in 
#        asyncio.get_event_loop().run_until_complete(run(args))
#
#
#class MemSuite:
#    def mem_list(self):
#        return [0] * 256
