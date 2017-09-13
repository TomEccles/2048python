import os
from collections import defaultdict


def mean(array):
    return sum(array) / len(array)


def var(array):
    return sum([x * x for x in array]) / len(array) - mean(array) ** 2


def mean_std(array):
    return (var(array) / len(array)) ** 0.5


class Analyser:
    def __init__(self, output_dir, prefix=""):
        results_files = [output_dir + "/" + s for s in os.listdir(output_dir) if s.startswith(prefix)]
        self.results = []
        for filename in results_files:
            with open(filename, "r") as file:
                turns = [int(line.split(" ")[0]) for line in file.readlines()]
                self.results.append((filename, turns))

    def get_stats(self):
        print("file, mean, max, std of mean")
        for (name, turns) in self.results:
            print((name, len(turns), mean(turns), max(turns), mean_std(turns)))


a = Analyser("./results/runs/full_run/output")

a.get_stats()


file_name = "./results/runs/full_run/output/search_results.txt"
res = defaultdict(list)
with open(file_name, "r") as file:
    lines = [[int(i) for i in line.split(" ")] for line in file.readlines()]
    for l in lines:
        t = (l[0], l[1])
        res[t].append(l[2])
keys = list(res.keys())
keys.sort()
for k in keys:
    print(k, mean(res[k]), mean_std(res[k]))
