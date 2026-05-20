#!/usr/bin/env python3
import numpy as np
import sys


def isMirrored(sq):
    return sq % 8 >= 4


# fmt: off
LAYOUT = [
    0,  1,  2,  3, 3, 2,  1,  0,
    4,  5,  6,  7, 7, 6,  5,  4,
    8,  8,  9,  9, 9, 9,  8,  8,
    10, 10, 11, 11, 11, 11, 10, 10,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
]
BUCKETS = max(LAYOUT) + 1
# fmt: on

HL = 1536
FT_SIZE = BUCKETS * 12 * 64 * HL
MERGED_FT_SIZE = BUCKETS * 11 * 64 * HL

net = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.int16)
ft = net[:FT_SIZE].reshape([BUCKETS, 12, 64, HL])
mergedFt = ft[:, :11, :, :].copy()

friendlyKing = 5
opponentKing = 11

for bucket in range(BUCKETS):
    for sq in range(64):
        if bucket == LAYOUT[sq] and not isMirrored(sq):
            continue
        mergedFt[bucket, friendlyKing, sq, :] = ft[bucket, opponentKing, sq, :]

mergedNet = np.concatenate((mergedFt.reshape(MERGED_FT_SIZE), net[FT_SIZE:]))
open("merged.nnue", "wb").write(mergedNet.tobytes())