import os

run_path = './runs/run1499148088'
output = run_path + "/output"

results_files = [output + "/" + s for s in os.listdir(output) if s.startswith("results")]


def number(filename):
    index = filename.find("iter")
    start = filename.find("_", index)
    end = filename.find(".", start + 1)
    return int(filename[start + 1:end])


results_files.sort(key=lambda name: number(name))
print(results_files)
results = []

for filename in results_files:
    with open(filename, "r") as file:
        turns = [int(line.split(" ")[0]) for line in file.readlines()]
        results.append(turns)

def average(array):
    return sum(array) / len(array)

means = [average(turns) for turns in results]
print(means)
split = int(len(means)/2)
print(average(means[:split]), average(means[split:]))
