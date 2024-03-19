from game2048 import *
from gen_algorithm import GameSolver


def main():
    # app = App()
    # app.run()
    f = open("demofile2.txt", "r")

    parsed_steps = None

    for line in f:
        steps = line.split(" ")
        parsed_steps = np.empty(len(steps) - 1)

        for i in range(0, len(parsed_steps)):
            parsed_steps[i] = int(float(steps[i]))

        parsed_steps = np.array(parsed_steps)

    f.close()

    gs = GameSolver(parsed_steps)
    steps = gs.generate_solution()
    f = open("demofile2.txt", "a")

    for step in steps:
        f.write(str(step) + " ")

    app = App()
    app.run(steps)

    f.close()

if __name__ == '__main__':
    main()
