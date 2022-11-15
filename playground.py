import collections
import csv
from ast import literal_eval

import numpy as np

if __name__ == "__main__":
    cosines = dict()

    cosines["a"] = 1
    cosines["b"] = 2
    cosines["c"] = 3
    cosines["d"] = 4
    cosines["e"] = 5
    cosines["f"] = 6
    cosines["g"] = 7
    cosines["h"] = 8
    cosines["i"] = 9
    cosines["j"] = 10
    results = sorted(cosines.items(), key=lambda kv: kv[1],
                              reverse=True)
    print([x[1] for x in sorted(results, key=lambda x: x[0])])

    relevance_unordered = [x[1] for x in sorted(results, key=lambda x: x[0])]