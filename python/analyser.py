import os


def mean(array):
    return sum(array) / len(array)


def var(array):
    return sum([x * x for x in array]) / len(array) - mean(array) ** 2


def mean_std(array):
    return (var(array) / len(array)) ** 0.5


class Analyser:
    def __init__(self, output_dir):
        results_files = [output_dir + "/" + s for s in os.listdir(output_dir)]
        self.results = []
        for filename in results_files:
            with open(filename, "r") as file:
                turns = [int(line.split(" ")[0]) for line in file.readlines()]
                self.results.append((filename, turns))

    def get_stats(self):
        print("file, mean, max, std of mean")
        for (name, turns) in self.results:
            print((name, len(turns), mean(turns), max(turns), mean_std(turns)))


a = Analyser("./runs/test2")
a.get_stats()
