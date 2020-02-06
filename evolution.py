import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import time

if len(sys.argv) != 7:
    print("python3 evolution.py <population_size> <binary/float> <mutation_prob> <crossover_prob> <max_iter> <task_number>")
    exit()

population_size = int(sys.argv[1])
unit_format = str(sys.argv[2])  # binary / float
mutation_probability = float(sys.argv[3])
crossover_probability = float(sys.argv[4])
max_iterations = int(sys.argv[5])
N = sys.argv[6]


def is_close(float1, float2, epsilon=1e-6):
    return abs(float1 - float2) <= epsilon


def f1(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f3(x_vector):
    a = 0
    for i in range(len(x_vector)):
        a += (x_vector[i] - i) ** 2
    return a


def f6(x_vector):
    temp_sum = 0.0
    for i in range(len(x_vector)):
        temp_sum += x_vector[i] ** 2
    a = np.sin(np.sqrt(temp_sum)) ** 2 - 0.5
    b = (1 + 0.001 * temp_sum) ** 2
    return 0.5 + a / b


def f7(x_vector):
    temp_sum = 0.0
    for i in range(len(x_vector)):
        temp_sum += x_vector[i] ** 2
    return temp_sum ** 0.25 * (1 + np.sin(50 * temp_sum ** 0.1))


# Enables us to not repeat calculations and keep track of number of evals
class GoalFunction:
    def __init__(self, function, start=None):
        self.f = function
        self.start = np.array(start)
        self.count = 0
        self.store = dict()

    def eval(self, x):
        if str(x) not in self.store.keys():
            self.store[str(x)] = self.f(x)
            self.count += 1
        return self.store[str(x)]

    def reset(self):
        self.count = 0
        self.store = dict()


class Chromosome:
    # binary precision 17 bits -> 10^-3 precision
    # binary precision 20 bits -> 10^-4 precision
    def __init__(self, format_type, dimension, lower=-50, upper=150, values=None, empty=False, binary_precision=20):
        self.function_value = None
        self.lower = lower
        self.upper = upper
        self.dimension = dimension
        self.format = format_type
        self.binary_precision = binary_precision
        self.binary_values = []  # used only with binary format

        if values is None:
            if empty is False:
                self.values = [random.uniform(self.lower, self.upper) for _ in range(dimension)]
            else:
                self.values = []
                for i in range(self.dimension):
                    self.values.append(0.0)
        else:
            self.values = values
        if self.format.startswith("b"):  # binary format
            for i in range(self.dimension):
                self.binary_values.append(self.to_binary(self.values[i]))

    def to_binary(self, x):
        try:
            b = np.floor(2 ** self.binary_precision - 1) / (self.upper - self.lower) * (x - self.lower)
        except ZeroDivisionError:
            self.lower = -50
            self.upper = 150
            b = np.floor(2 ** self.binary_precision - 1) / (self.upper - self.lower) * (x - self.lower)

        str_b, list_b = format(int(b), "b").zfill(self.binary_precision), []
        for i in range(len(str_b)):
            list_b.append(str_b[i])
        return list_b

    def to_float(self, b):
        if type(b) is list:
            a = ""
            for item in b:
                a += str(item)
            b = a
        b = int(str(b), 2)
        return self.lower + b * (self.upper - self.lower) / (2 ** self.binary_precision - 1)

    def set(self, index, value, binary=False):
        self.values[index] = value
        if binary is True:
            self.binary_values[index] = self.to_binary(self.values[index])

    def __str__(self):
        s = ""
        for item in self.values:
            s += str(item) + "  "
        if self.format.startswith("b"):
            s += "\nBinary: "
            for item in self.binary_values:
                for x in item:
                    s += str(x)
                s += "  "
        s += "\nFunction value: " + str(self.function_value)
        return s

    def __eq__(self, other):
        equal = True
        for i in range(len(self.values)):
            if is_close(self.values[i], other.values[i]) is False:
                equal = False
                break
        return equal


class Population:
    def __init__(self, size, format_type, dimension, lower=-50, upper=150, empty=False):
        self.chromosomes = []
        self.iterator = 0
        self.size = size
        self.format = format_type
        self.lower = lower
        self.upper = upper
        self.dimension = dimension

        if empty is False:
            for i in range(size):
                self.chromosomes.append(Chromosome(format_type, dimension, lower=lower, upper=upper, empty=False))
        else:
            for i in range(size):
                self.chromosomes.append(Chromosome(format_type, dimension, lower=lower, upper=upper, empty=True))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator >= self.size:
            self.iterator = 0
            raise StopIteration
        else:
            self.iterator += 1
            return self.chromosomes[self.iterator - 1]

    def __str__(self):
        s = ""
        for item in self.chromosomes:
            s += str(item) + "\n"
        return s

    def __len__(self):
        return self.size

    def evaluate(self, function):
        for c in self.chromosomes:
            c.function_value = function.eval(c.values)

    def sort(self):
        self.chromosomes.sort(key=lambda x: x.function_value)


class GA:
    def __init__(self, population, crossover_chance, mutation_chance, k, max_iter, function, precision=1e-4,
                 evalcount=None):
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        self.population = population
        self.max_iter = max_iter
        self.k = k
        self.precision = precision
        self.function = function
        if self.population.format.startswith('b'):
            self.binbool = True
        else:
            self.binbool = False
        self.evalcount = evalcount

    '''
        CROSSOVER METHODS        
    '''

    # only for floating point chromosomes
    # changes random (x-th) value to a value between parents' values
    def arithmetic_crossover(self, parent1, parent2):
        child = Chromosome(parent1.format, parent1.dimension, empty=True)
        preferred_parent = random.randint(0, 1)  # one which values is taken if crossover does not happen

        for x in range(child.dimension):
            if random.uniform(0, 1) <= self.crossover_chance:
                r = random.uniform(0, 1)
                child.set(x, r * parent1.values[x] + (1 - r) * parent2.values[x])
            else:
                if preferred_parent == 1:
                    child.set(x, parent1.values[x])
                else:
                    child.set(x, parent2.values[x])
        return child

    # only for floating point chromosomes
    # changes random (x-th) value to a value around or between parents' values
    def heuristic_crossover(self, parent1, parent2):
        child = Chromosome(parent1.format, parent1.dimension, empty=True)
        preferred_parent = random.randint(0, 1)

        for i in range(child.dimension):
            if random.uniform(0, 1) <= self.crossover_chance:
                r = random.uniform(0, 1)
                child.set(i, r * (parent1.values[i] - parent2.values[i]) + parent2.values[i])

                while child.values[i] > child.upper or child.values[i] < child.lower:  # check bounds
                    r = random.uniform(0, 1)
                    child.set(i, r * (parent1.values[i] - parent2.values[i]) + parent2.values[i])
            else:
                if preferred_parent == 1:
                    child.set(i, parent1.values[i])
                else:
                    child.set(i, parent2.values[i])
        return child

    # swaps values between parents
    # used for both formats
    def simple_shuffle(self, parent1, parent2):
        child = Chromosome(parent1.format, parent1.dimension)
        if random.randint(0, 1) == 1:  # allows more unique results
            parent1, parent2 = parent2, parent1

        for i in range(child.dimension):
            if random.uniform(0, 1) <= self.crossover_chance:
                child.set(i, parent1.values[i], binary=self.binbool)
            else:
                child.set(i, parent2.values[i], binary=self.binbool)
        return child

    # each bit is taken randomly from parent1 or parent2 (only one value)
    # used only for binary format
    def single_point_crossover(self, parent1, parent2):
        child = Chromosome(parent1.format, parent1.dimension)
        if random.randint(0, 1) == 1:  # expands scope
            parent1, parent2 = parent2, parent1

        for i in range(child.dimension):
            if random.uniform(0, 1) <= self.crossover_chance:
                point = random.randint(0, child.binary_precision - 1)  # choose separator
                for j in range(point):
                    child.binary_values[i][j] = parent1.binary_values[i][j]
                for j in range(point, child.binary_precision):
                    child.binary_values[i][j] = parent2.binary_values[i][j]
                for j in range(child.dimension):
                    child.values[j] = child.to_float(child.binary_values[j])
            else:
                child.set(i, parent1.values[i], binary=True)
        return child

    '''
        MUTATION OPERATORS
    '''
    # set to upper or lower bound
    # used for both formats
    def boundary_set(self, c):
        for position in range(c.dimension):
            if random.uniform(0, 1) <= self.mutation_chance:
                c.set(position, c.upper, binary=self.binbool)

    # set to random value in bounds
    # used for both formats
    def random_value(self, c):
        for position in range(c.dimension):
            if random.uniform(0, 1) <= self.mutation_chance:
                c.set(position, random.uniform(c.lower, c.upper), binary=self.binbool)

    '''
        SELECTION ALGORITHM IS K-TOURNAMENT
    '''
    # p is population from which k tournament plays on
    # returns 2 best out of k possible
    def choose_parents_ktournament(self):
        possible_parents = [random.choice(self.population.chromosomes) for i in range(self.k)]
        possible_parents.sort(key=lambda x: x.function_value)
        return possible_parents[0], possible_parents[1]

    # population must be init when algorithm is run
    def run(self):
        start = time.time()
        if self.binbool is True:
            self.precision = 1e-4
        self.population.evaluate(self.function)
        self.population.sort()
        the_best = self.population.chromosomes[0]

        success, i = False, 0
        #  for i in range(self.max_iter):
        while True:
            i += 1
            next_population = Population(self.population.size, self.population.format, self.population.dimension,
                                         lower=self.population.lower, upper=self.population.upper, empty=True)
            # elitism
            elite_count = 0
            next_population.chromosomes[elite_count] = self.population.chromosomes[elite_count]
            elite_count += 1
            # adding more elites is very bad
            # next_population.chromosomes[elite_count] = self.population.chromosomes[elite_count]
            # elite_count += 1

            for j in range(elite_count, self.population.size):
                parent1, parent2 = self.choose_parents_ktournament()
                # choose crossover operator randomly
                decision = random.randint(0, 2)

                if self.population.format.startswith("f"):  # float format has 3 operators
                    if decision == 0:
                        child = self.arithmetic_crossover(parent1, parent2)
                    elif decision == 1:
                        child = self.heuristic_crossover(parent1, parent2)
                    elif decision == 2:
                        child = self.simple_shuffle(parent1, parent2)
                else:  # binary format does not use arithmetic, only 2 operators
                    if decision == 0:
                        child = self.simple_shuffle(parent1, parent2)
                    else:
                        child = self.single_point_crossover(parent1, parent2)

                # choose mutation operator randomly
                decision = random.randint(0, 1)
                if decision == 1:
                    self.boundary_set(child)
                else:
                    self.random_value(child)
                next_population.chromosomes[j] = child

            self.population = next_population
            self.population.evaluate(self.function)
            self.population.sort()
            if the_best != self.population.chromosomes[0]:
                print("Generation ", i, ": best chromosome: ", self.population.chromosomes[0])
                print("Number of function evaluations: ", self.function.count)
            the_best = self.population.chromosomes[0]

            # Exit conditions
            # evalcount overrides all if not None
            if self.evalcount is not None:
                if self.function.count >= self.evalcount:
                    if abs(self.population.chromosomes[0].function_value) <= self.precision:
                        success = True
                    break

            # precision based exit condition
            if abs(self.population.chromosomes[0].function_value) <= self.precision and self.evalcount is None:
                success = True
                end = time.time()
                print("Solution found before max iterations in {} seconds!!".format(end - start))
                break

            # max iterations exit condition
            if self.evalcount is None and i >= self.max_iter:
                break

        if success is not True:
            print("Failed to find the solution in {} iterations!!".format(self.max_iter))
            # input("Press Enter to continue...")
            return False, self.population
        else:
            # input("Press Enter to continue...")
            return True, self.population


def task1():
    gf1 = GoalFunction(f1)
    gf3 = GoalFunction(f3)
    gf6 = GoalFunction(f6)
    gf7 = GoalFunction(f7)

    population1 = Population(population_size, unit_format, 2, lower=-50, upper=150)
    population3 = Population(population_size, unit_format, 5, lower=-50, upper=150)
    population6 = Population(population_size, unit_format, 2, lower=-50, upper=150)
    population7 = Population(population_size, unit_format, 2, lower=-50, upper=150)

    GA1 = GA(population1, crossover_probability, mutation_probability, 3, max_iterations, gf1)
    GA3 = GA(population3, crossover_probability, mutation_probability, 3, max_iterations, gf3)
    GA6 = GA(population6, crossover_probability, mutation_probability, 3, max_iterations, gf6)
    GA7 = GA(population7, crossover_probability, mutation_probability, 3, max_iterations, gf7)

    flag1, pop1 = GA1.run()
    input("Press Enter to continue...")
    flag3, pop3 = GA3.run()
    input("Press Enter to continue...")
    flag6, pop6 = GA6.run()
    input("Press Enter to continue...")
    flag7, pop7 = GA7.run()

    print("f1 ", flag1)
    print("f3 ", flag3)
    print("f6 ", flag6)
    print("f7 ", flag7)


def task2():
    flags, dims = [], [1, 3, 5, 7, 9]
    for dimension in dims:
        print("Dimensionality: ", dimension)
        gf6 = GoalFunction(f6)
        gf7 = GoalFunction(f7)
        population6 = Population(population_size, unit_format, dimension, lower=-50, upper=150)
        population7 = Population(population_size, unit_format, dimension, lower=-50, upper=150)
        GA6 = GA(population6, crossover_probability, mutation_probability, 3, max_iterations, gf6)
        GA7 = GA(population7, crossover_probability, mutation_probability, 3, max_iterations, gf7)
        flag6, pop6 = GA6.run()
        flag7, pop7 = GA7.run()
        flags.append([flag6, flag7])
        print("Dimensionality: ", dimension, "DONE!")
    for f, d in zip(flags, dims):
        print(d, f)


def task3():
    number_of_rounds, ecount = 10, 10000  # function evaluations maximum
    for d in [3, 6]:
        results_float, results_binary = [], []
        for i in range(number_of_rounds):
            gf3 = GoalFunction(f7)  # hot fix to f7
            gf6 = GoalFunction(f6)
            population3 = Population(population_size, "float", d, lower=-50, upper=150)
            population6 = Population(population_size, "float", d, lower=-50, upper=150)
            GA3 = GA(population3, crossover_probability, mutation_probability, 3, max_iterations, gf3, evalcount=ecount)
            GA6 = GA(population6, crossover_probability, mutation_probability, 3, max_iterations, gf6, evalcount=ecount)
            flag3, pop3 = GA3.run()
            flag6, pop6 = GA6.run()
            results_float.append([flag3, flag6, pop3.chromosomes[0].function_value, pop6.chromosomes[0].function_value])

        for i in range(number_of_rounds):
            gf3 = GoalFunction(f7)  # hot fix to f7
            gf6 = GoalFunction(f6)
            population3 = Population(population_size, "binary", d, lower=-50, upper=150)
            population6 = Population(population_size, "binary", d, lower=-50, upper=150)
            GA3 = GA(population3, crossover_probability, mutation_probability, 3, max_iterations, gf3, evalcount=ecount)
            GA6 = GA(population6, crossover_probability, mutation_probability, 3, max_iterations, gf6, evalcount=ecount)
            flag3, pop3 = GA3.run()
            flag6, pop6 = GA6.run()
            results_binary.append(
                [flag3, flag6, pop3.chromosomes[0].function_value, pop6.chromosomes[0].function_value])

        bin_count, float_count = [0, 0], [0, 0]
        for b, f in zip(results_binary, results_float):
            if b[0] is True:
                bin_count[0] += 1
            if b[1] is True:
                bin_count[1] += 1
            if f[0] is True:
                float_count[0] += 1
            if f[1] is True:
                float_count[1] += 1
        print("Dimension ", d, " results:")
        print("Team Float scored {}/{} on f3. Median: {}".format(float_count[0], number_of_rounds,
                                                                 np.median([row[2] for row in results_float])))
        print("Team Binary scored {}/{} on f3. Median: {}".format(bin_count[0], number_of_rounds,
                                                                  np.median([row[2] for row in results_binary])))
        print("Team Float scored {}/{} on f6. Median: {}".format(float_count[1], number_of_rounds,
                                                                 np.median([row[3] for row in results_float])))
        print("Team Binary scored {}/{} on f6. Median: {}".format(bin_count[1], number_of_rounds,
                                                                  np.median([row[3] for row in results_binary])))
        input("ENTER to continue")


def task4():
    ecount = 2000  # function evaluations for each run
    number_of_rounds, results = 30, []
    pop_medians, mut_medians = [], []
    popsize_results, index = [], 0
    for s in [30, 50, 100, 200]:
        popsize_results.append([])
        for i in range(number_of_rounds):
            gf6 = GoalFunction(f6)
            population6 = Population(s, unit_format, 2, lower=-50, upper=150)
            GA6 = GA(population6, crossover_probability, mutation_probability, 3, max_iterations, gf6, evalcount=ecount)
            flag6, pop6 = GA6.run()
            popsize_results[index].append(pop6.chromosomes[0].function_value)
        pop_medians.append(np.median(popsize_results[index]))
        index += 1
    plt.boxplot(popsize_results)
    plt.show()

    mut_results, index = [], 0
    for m in [0.1, 0.3, 0.6, 0.9]:
        mut_results.append([])
        for i in range(number_of_rounds):
            gf6 = GoalFunction(f6)
            population6 = Population(population_size, unit_format, 2, lower=-50, upper=150)
            GA6 = GA(population6, crossover_probability, m, 3, max_iterations, gf6, evalcount=ecount)
            flag6, pop6 = GA6.run()
            mut_results[index].append(pop6.chromosomes[0].function_value)

        mut_medians.append(np.median(mut_results[index]))
        index += 1
    plt.boxplot (mut_results)
    plt.show()

    i = 0
    for m1 in mut_medians:
        i += 1
        print("Mutation Experiment {} median: {}".format(i, m1))
    i = 0
    for m2 in pop_medians:
        i += 1
        print("Population Size Experiment {} median: {}".format(i, m2))


def task5():
    ecount = 30000
    for kk in [3, 4, 5, 10]:
        gf6 = GoalFunction(f6)
        gf7 = GoalFunction(f7)
        population6 = Population(population_size, unit_format, 2, lower=-50, upper=150)
        population7 = Population(population_size, unit_format, 2, lower=-50, upper=150)
        GA6 = GA(population6, crossover_probability, mutation_probability, kk, max_iterations, gf6, evalcount=ecount)
        GA7 = GA(population7, crossover_probability, mutation_probability, kk, max_iterations, gf7, evalcount=ecount)
        GA6.run()
        GA7.run()


locals()["task" + str(N)]()
