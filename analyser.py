import os
output = "./runs/test2"
results_files = [output + "/" + s for s in os.listdir(output)]



def number(filename):
    index = filename.find("iter")
    start = filename.find("_", index)
    end = filename.find(".", start + 1)
    return int(filename[start + 1:end])

#run_path = "./runs/" + [s for s in os.listdir("./runs") if s.startswith("run")][-1]
#output = run_path + "/output"
#results_files = [output + "/" + s for s in os.listdir(output) if s.startswith("results")]
#results_files.sort(key=lambda name: number(name))
print(results_files)
results = []

for filename in results_files:
    with open(filename, "r") as file:
        turns = [int(line.split(" ")[0]) for line in file.readlines()]
        results.append(turns)

def mean(array):
    return sum(array) / len(array)

def var(array):
    return sum([x*x for x in array]) / len(array) - mean(array)**2

def mean_std(array):
    return (var(array)/len(array)) ** 0.5

print("file, mean, max, std of mean")
for (turns, name) in zip(results, results_files):
    print((name, len(turns), mean(turns), max(turns), mean_std(turns)))
#split = int(len(means)/2)
#print(average(means[:split]), average(means[split:]))
