# Genetic-Algorithm
Genetic algorithm proven to be very effective on optimization problems with many local optima. Supports binary and floating point representations. Mimicks natural selection to find global optimum.

[+] Comes with 5 prepared test problems (check pdf for functions, text is in Croatian though)
[+] Shows boxplots of the population which is useful for testing various configurations. 

# Usage
$ python3 evolution.py 
python3 evolution.py <population_size> <binary/float> <mutation_prob> <crossover_prob> <max_iter> <task_number>

# Example
Finding minimum of Schaffer function and slightly modified Schaffer function (see pdf) in 3 and 6 dimensions. 

$ python3 evolution.py 50 binary 0.05 0.3 1000 3
Generation  1 : best chromosome:  -26.333986355959915  8.53224010926077  150  
Function value: 7.775098620854167e-06
Number of function evaluations:  84
Generation  26 : best chromosome:  -16.82989631563775  -21.938541737529626  150  
Function value: 6.700945454162722e-08
Number of function evaluations:  812
Generation  51 : best chromosome:  -16.82989631563775  -21.938541737529626  149.99921254245552  
Function value: 7.391996745966489e-09
Number of function evaluations:  1475
Generation  52 : best chromosome:  -16.82989631563775  -21.93493307238452  150.0  
Function value: 7.702251733671443e-10
Number of function evaluations:  1503
Generation  58 : best chromosome:  -16.82989631563775  -21.938541737529626  149.99937610688391  
Function value: 2.0591014569824416e-10
Number of function evaluations:  1670
Generation  62 : best chromosome:  -16.82989631563775  -21.93493307238452  149.99993715585933  
Function value: 5.347461307748055e-14
Number of function evaluations:  1774
Generation  100 : best chromosome:  -16.82989631563775  -21.934945107685653  149.99993470764582  
Function value: 4.113431766063901e-15
Number of function evaluations:  2667
Generation  102 : best chromosome:  -16.82989631563775  -21.935423525001877  149.99986484940956  
Function value: 0.0
(...)
