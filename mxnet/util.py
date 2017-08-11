from __future__ import print_function

def decideInterval(d_size, batch_size):
    assert d_size >= batch_size, "batch size cannot larger than dataset size"
    interval = d_size / batch_size
    buckets = [10000, 100, 1]
    for bucket in buckets:
        if interval >= bucket:
            return bucket