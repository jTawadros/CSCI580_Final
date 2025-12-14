import os
import torch
import matplotlib.pyplot as plt

import fp_model
import fp_work

images, labels, groups = fp_work.ProjectDataLoader(print_loaded=True)

def run(name, batch_len, model, optimizer, max_epoch, use_augmented_data=False):
	print("### " + name + " ###")
	RESULTS_FILE = "benchmark_results"

	try:
		os.mkdir(RESULTS_FILE)
	except FileExistsError:
		pass

	os.mkdir(f'{RESULTS_FILE}/{name}')

	losses = fp_work.train(batch_len, model, optimizer, max_epoch, use_augmented_data)

	# Plot loss over epochs
	plt.plot(losses, marker='o')
	plt.xticks(range(max_epoch), range(1, max_epoch + 1))
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Training Loss Over {max_epoch} Epochs')
	plt.savefig(f'{RESULTS_FILE}/{name}/loss_over_{max_epoch}_epochs.png')
	plt.close()

	outputsA, lossA, correctA, totalA, per_digit_correctA, per_digit_totalA, true_positiveA, false_positiveA, false_negativeA = fp_work.testMNIST(model, 50)
	outputsB, lossB, correctB, totalB,_,_, per_digit_correctB, per_digit_totalB, group_correctB, group_totalB,_,_ , true_positiveB, false_positiveB, false_negativeB = fp_work.testImages(model, images, labels, groups)
	# F1 Score
	def f1_score(true_positive, false_positive, false_negative):
		precision = true_positive/(true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
		recall = true_positive/(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
		return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
	# F1 for MNIST
	mnist_f1_scores = []
	class_f1_scores = []
	for i in range(10):
		mnist_f1_scores.append(f1_score(true_positiveA[i], false_positiveA[i], false_negativeA[i]))
		class_f1_scores.append(f1_score(true_positiveB[i], false_positiveB[i], false_negativeB[i]))

	print("MNIST F1 Scores per Digit: " + str(mnist_f1_scores))
	print("Class Digits F1 Scores per Digit: " + str(class_f1_scores))
	#$ plot mnist F1 Scores
	plt.figure(figsize=(8,5))
	plt.bar(range(10), mnist_f1_scores)
	plt.xlabel("Digit")
	plt.ylabel("F1 Score")
	plt.title("MNIST F1 Scores per Digit")
	plt.ylim(0,1)
	plt.savefig(f'{RESULTS_FILE}/{name}/mnist_f1_scores.png')
	plt.close()
	# plot class digits F1 Scores
	plt.figure(figsize=(8,5))
	plt.bar(range(10), class_f1_scores)
	plt.xlabel("Digit")
	plt.ylabel("F1 Score")
	plt.title("Class Digits F1 Scores per Digit")
	plt.ylim(0,1)
	plt.savefig(f'{RESULTS_FILE}/{name}/class_f1_scores.png')
	plt.close()

	# Calculate per-group accuracy
	group_ids = sorted(group_totalB.keys())

	group_accuracy = []
	for g in group_ids:
		acc = group_correctB[g] / group_totalB[g] if group_totalB[g] > 0 else 0
		group_accuracy.append(acc)

	plt.figure(figsize=(8,5))
	plt.bar(group_ids, group_accuracy)
	plt.xlabel("Group ID")
	plt.ylabel("Accuracy")
	plt.title("Class Digits Group Accuracy")
	plt.ylim(0,1)
	plt.xticks(group_ids, group_ids)
	plt.grid(axis='y', linestyle='--', alpha=0.4)
	plt.savefig(f'{RESULTS_FILE}/{name}/group_accuracy.png')
	plt.close()

	# Calculate per-digit accuracy for MNIST and Class Digits
	digit_accuracyA = []
	for i in range(10):
		if per_digit_totalA[i] > 0:
			accuracy = per_digit_correctA[i] / per_digit_totalA[i]
		else:
			accuracy = 0
		digit_accuracyA.append(accuracy)
	digit_accuracyB = []
	for i in range(10):
		if per_digit_totalB[i] > 0:
			accuracy = per_digit_correctB[i] / per_digit_totalB[i]
		else:
			accuracy = 0
		digit_accuracyB.append(accuracy)

	# Plot per-digit accuracy for MNIST and Class Digits
	plt.bar(range(10), digit_accuracyA)
	plt.xlabel("Digit")
	plt.ylabel("Accuracy")
	plt.title("MNIST Accuracy")
	plt.ylim(0, 1)
	plt.savefig(f'{RESULTS_FILE}/{name}/mnist_per_digit.png')
	plt.close()

	plt.bar(range(10), digit_accuracyB)
	plt.xlabel("Digit")
	plt.ylabel("Accuracy")
	plt.title("Class Digits Accuracy")
	plt.ylim(0, 1)
	plt.savefig(f'{RESULTS_FILE}/{name}/class_per_digit.png')
	plt.close()

	# Return overall results
	return correctA/totalA, correctB/totalB

def best(name_array, mnist_correct_array, class_correct_array):
	mnist_max_id = 0
	class_max_id = 0
	added_max_id = 0

	for i in range(1, len(name_array)):
		if mnist_correct_array[mnist_max_id] < mnist_correct_array[i]:
			mnist_max_id = i
		if class_correct_array[class_max_id] < class_correct_array[i]:
			class_max_id = i
		if mnist_correct_array[added_max_id] + class_correct_array[added_max_id] < mnist_correct_array[i] + class_correct_array[i]:
			added_max_id = i

	print(f'Best Model for MNIST: {name_array[mnist_max_id]} ({mnist_correct_array[mnist_max_id]})')
	print(f'Best Model for Class: {name_array[class_max_id]} ({class_correct_array[class_max_id]})')
	print(f'Best Model Overall: {name_array[added_max_id]} ({mnist_correct_array[added_max_id]}, {class_correct_array[added_max_id]})')