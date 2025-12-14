import math
import fp_benchmark
import fp_model
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 5
BATCH_SIZE = 96
NEURON_MULT = 30
NEURON_BASE = 10
PER_LAYER_ITER = 4
NEURON_MAX = NEURON_BASE + NEURON_MULT * (PER_LAYER_ITER - 1)

name_array = []
mnist_correct_array = []
class_correct_array = []
for middle_alg in [{"name":"ReLU", "what":nn.ReLU()}, {"name":"LeakyReLU", "what":nn.LeakyReLU(0.1)}]:
	for layer_num in range(1,3):
		lA = NEURON_BASE
		lB = NEURON_BASE
		for i in range(1, int(math.pow(PER_LAYER_ITER, layer_num)+1)):
			layering = [lA]
			if layer_num == 2:
				layering += [lB]
			for learn_rate in [0.01, 0.1]:
				for optimizer in [{"name": "SGD", "what":optim.SGD}, {"name": "Adam", "what":optim.Adam}]:
					model = fp_model.NPlus1LenModel(middle_alg["what"], layering)
					namestr = middle_alg["name"] + '-' + str(layering) + '-' + optimizer["name"] + '-' + str(learn_rate)
					mnist_correct, class_correct = fp_benchmark.run(namestr, BATCH_SIZE, model, optimizer["what"](model.parameters(), lr = learn_rate), EPOCHS)
					print("MNIST Proportion Correct: " + str(mnist_correct) + "\nClass Proportion Correct: " + str(class_correct))
					name_array += [namestr]
					mnist_correct_array += [mnist_correct]
					class_correct_array += [class_correct]
			lA += NEURON_MULT
			if (lA > NEURON_MAX):
				lB += NEURON_MULT
				lA = NEURON_BASE

results = []
for i in range(len(name_array)):
	results.append((name_array[i], class_correct_array[i]))
	results.sort(key=lambda x: x[1], reverse=True)
	top5 = results[:5]
	names = [top5[i][0] for i in range(len(top5))]
	scores = [top5[i][1] for i in range(len(top5))]
	plt.figure(figsize=(10,5))
	plt.bar(names, scores)
	plt.ylabel('Class Proportion Correct')
	plt.xlabel('Model Configuration')
	plt.ylim(0, 1)
	plt.xticks(rotation=45, ha='right')
	plt.savefig('top5_models.png', bbox_inches='tight')
	plt.show()


fp_benchmark.best(name_array, mnist_correct_array, class_correct_array)
