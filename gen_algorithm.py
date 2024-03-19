import numpy as np
from game2048 import Board
from numpy import ndarray
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import time


class Individual:
    def __init__(self, steps):
        self.steps = steps
        self.fitness_value = -1

    @classmethod
    def generate_random_individual(cls, steps_seed: ndarray):
        if steps_seed.any():
            steps = np.empty(len(steps_seed))

            for (i, step_seed) in enumerate(steps_seed):
                one_prob = 0.05
                probs = [one_prob, one_prob, one_prob, one_prob]
                probs[int(step_seed)] = 1 - (3 * one_prob)
                steps[i] = np.random.choice(4, p=probs)
                # steps[i] = step_seed
        else:
            generatable_count = GameSolver.CHROMOSOME_COUNT - 1
            steps = np.concatenate((np.array([2]), np.random.choice(4, generatable_count, p=[0.05, 0.40, 0.1, 0.45])))

        return cls(steps)

    def fitness(self):
        board = Board(4)
        steps_list: list = self.steps.tolist()

        while steps_list and not board.completed():
            curr_step = steps_list.pop(0)
            board.move(curr_step)

        # zeros = np.count_nonzero(board.get_board() == 0)
        # self.fitness_value = zeros
        # return zeros

        board_length = len(board.get_board())
        score = 0
        # max_val = -1
        #

        for i in range(0, board_length):
            for j in range(0, board_length):
                # val = board.get_val(i, j)
                score += board.get_val(i, j)

                # if val > max_val:
                #     max_val = val
        #
        self.fitness_value = score
        return self.fitness_value

    def crossover(self, other: 'Individual', split_point):
        self.steps = np.concatenate([self.steps[:split_point], other.steps[split_point:]])

    def mutate(self, mutation_count):
        mutation_count = int(mutation_count)
        mutation_indeces = np.random.randint(0, GameSolver.CHROMOSOME_COUNT, mutation_count)
        mutation_values = np.random.choice(4, size=mutation_count, p=[0.00, 0.40, 0.1, 0.45])
        # mutation_values = np.random.choice(4, size=mutation_count)

        for i in range(0, mutation_count):
            self.steps[mutation_indeces[i]] = mutation_values[i]

    def get_fitness_value(self):
        return self.fitness_value

    def get_steps(self):
        return self.steps


def mutate_remainders(remainder, mutation_count):
    # max_mut_val = int(GameSolver.CHROMOSOME_COUNT * GameSolver.MUTATION_RATE)
    # rand_nr = np.random.rand(0, 10)
    #
    # if rand_nr < max_mut_val:
    remainder.mutate(mutation_count)


def evaluate_fitness(individual):
    individual.fitness()


class GameSolver:
    GENERATIONS = 10
    POPULATION_SIZE = 200
    CHROMOSOME_COUNT = 150
    MUTATION_RATE = 0.3
    ELITISM_RATE = 0.1

    def __init__(self, steps=None):
        if steps is None:
            steps = []

        self.individuals = np.empty(self.POPULATION_SIZE, dtype=Individual)
        self.steps_seed = steps

    def generate_solution(self) -> ndarray:
        self.create_population_async()
        self.run_generations()

        return self.get_best_solution().get_steps()

    def create_population(self, index):
        self.individuals[index] = Individual.generate_random_individual(self.steps_seed)

    def create_population_async(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.create_population, i) for i in range(0, self.POPULATION_SIZE)]
            wait(futures)

    def evaluate_fitnesses_async(self):
        start = time.time()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_fitness, individual) for individual in self.individuals]
            wait(futures)

        end = time.time()
        print(f'Parallel: {end - start}')

    def evaluate_fitnesses(self):
        start = time.time()

        for individual in self.individuals:
            individual.fitness()

        end = time.time()
        print(f'Linear: {end - start}')

    def run_generations(self):
        for i in range(0, self.GENERATIONS):
            # Calculate fitness value for individual
            self.evaluate_fitnesses_async()

            elites, remainders = self.split_individuals()
            elites: ndarray
            remainders: ndarray

            self.cross_remainders(remainders)
            self.mutate_remainders_async(remainders)

            self.individuals = np.concatenate((elites, remainders), axis=None)

            last_elite_item_index = int(self.POPULATION_SIZE * self.ELITISM_RATE - 1)
            best_elem: Individual = elites[last_elite_item_index]
            print(f'Generation {i} finished! Best element fitness: {best_elem.get_fitness_value()}')

    def elite_count(self) -> int:
        return int(self.POPULATION_SIZE * self.ELITISM_RATE)

    def split_individuals(self):
        scores = np.array([ind.get_fitness_value() for ind in self.individuals])

        for (i, individual) in enumerate(self.individuals):
            scores[i] = individual.get_fitness_value()

        k = self.elite_count()

        sorted_indeces = np.argsort([individual.get_fitness_value() for individual in self.individuals])
        self.individuals = self.individuals[sorted_indeces]

        elites = self.individuals[-k:]
        remainders = self.individuals[:-k]

        return elites, remainders

    def cross_remainders(self, remainders: ndarray):
        remainders_len = len(remainders)

        for i in range(remainders_len):
            remainder_to_cross_with = i

            while remainder_to_cross_with == i:
                remainder_to_cross_with = np.random.randint(0, remainders_len)

            split_point = np.random.randint(0, self.CHROMOSOME_COUNT - 1) + 1
            remainders[i].crossover(remainders[remainder_to_cross_with], split_point)

    def mutate_remainders_async(self, remainders: ndarray):
        mutation_count = self.CHROMOSOME_COUNT * self.MUTATION_RATE

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(mutate_remainders, remainder, mutation_count) for remainder in remainders]
            wait(futures)

    def get_best_solution(self):
        max_score = -2
        solution = None

        for individual in self.individuals:
            if individual.get_fitness_value() > max_score:
                max_score = individual.get_fitness_value()
                solution = individual

        return solution
