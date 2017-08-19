class IndentedPrinter:
    def __init__(self, depth):
        self.depth = depth

    def print(self, string):
        print(" " * self.depth, sep="", end="")
        print(string)
