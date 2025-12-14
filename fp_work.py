import torch
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

IMAGE_IN = 784

def ProjectDataLoader(print_loaded=False):
	CLASS_DIGITS_DIR = "class_digits"

	if print_loaded == True:
		print ("--- Loading Data From ./" + CLASS_DIGITS_DIR + " ---")

	raw_images = []
	images = []
	labels = np.array([])
	groups = np.array([])
	process = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5, inplace=False)])

	for member in range(1, 5):
		for group in range(1, 11):
			for digit in range(0, 10):
				path_str = CLASS_DIGITS_DIR + '/' + str(digit) + '-' + str(group) + '-' + str(member) + ".png"

				try:
					raw_image = Image.open(path_str).convert('L')
				except FileNotFoundError:
					#if print_loaded == True:
					#	print("Tried but failed to find " + path_str + '.')
					continue
				image = process(raw_image)
				group_id = group
				groups = np.append(groups, group_id)

				raw_images.append(raw_image)
				images.append(image)
				labels = np.append(labels, digit)
				
				if print_loaded == True:
					print("Loaded in " + path_str + '.')

	raw_images = np.array(raw_images)
	images = torch.Tensor(np.array(images))
	labels = torch.Tensor(labels).long()

	if print_loaded == True:
			print ("--- Finished Loading Data From ./" + CLASS_DIGITS_DIR + " ---")

	print("Data: " + str(raw_images.shape))
	print("Transformed Data: " + str(images.shape))
	print("Labels: " + str(labels.shape))

	return images, labels, torch.Tensor(groups).long()

def train(batch_len, model, optimizer, max_epoch, use_augmented_data=False):
	std_transform = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Normalize((0.5,), (0.5,))
		])
	augmented_transform = transforms.Compose([
		transforms.RandomRotation(10),
		transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.2)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
		])
	if use_augmented_data:
		trainingData = torch.utils.data.ConcatDataset([datasets.MNIST('.', download=True, train=True, transform=std_transform), datasets.MNIST('.', download=True, train=True, transform=augmented_transform)])
		print("Using augmented data for training.")
	else:
		trainingData = datasets.MNIST('.', download=True, train=True, transform=std_transform)
	trainingDataLoader = torch.utils.data.DataLoader(dataset=trainingData, batch_size=batch_len, shuffle=True)

	print("--- Began Training ---")
	print("Batch Length --", batch_len)
	print("Model --", model)
	print("Optimizer --", optimizer)
	print("Max Epoch --", max_epoch)

	avgLossesPerEpoch = []
	epoch = 0
	while (epoch < max_epoch):
		avg_loss = 0
		num_batches = 0
		for images, labels in iter(trainingDataLoader):
			optimizer.zero_grad()
			images.resize_(batch_len, IMAGE_IN)
			output, loss = model.forward(images, labels)
			loss.backward()
			optimizer.step()
			num_batches += 1
			avg_loss += loss.item()
		avg_loss /= num_batches
		avgLossesPerEpoch.append(avg_loss)
		print("--- Finished Epoch", epoch + 1, "---")
		print("Average Loss:", avg_loss)
		epoch += 1
	print("--- Finished Training ---")
	#model.printParam()

	return avgLossesPerEpoch

# =============================== Source Credits =======================================
# Pytorch documentation and tutorials provides a baseline for testing models on images.
# The following code is adapted from the tutorial listed below.
# Modifications:
	# Firstly, instead of loading in the CIFAR10 dataset, this code will load in either MNIST or our class digits.
	# Secondly, this code is adapted to work with our NPlus1LenModel structure.
	# Some other small variations include the fact that a tuple is returned from our model's forward function,
	# so we need to index it accordingly. While the tutorial uses a 'class' structure for identifying the network, 
	#.this code uses a functional structure uses the model defined in fp_model.py.
# Link: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#testing-the-network-on-the-test-data
# =================================================================================

def testImages(model, images, labels, groupIds):
	images = images.view(images.size(0), IMAGE_IN)
	per_digit_correct = [0] * 10
	per_digit_total = [0] * 10
	group_correct = {}
	group_total = {}
	total_predicted = []
	total_labels = []
	true_positive = [0] * 10
	false_positive = [0] * 10
	false_negative = [0] * 10
	with torch.no_grad():
		outputs,loss = model(images, labels)
		_, predicted = torch.max(outputs.data, 1)
		# per_digit values in this function are for class digits testing
		# MNIST testing has its own per_digit tracking in testMNIST function
		for i in range(len(predicted)):
			p = predicted[i]
			l = labels[i]
			g = groupIds[i].item()
			total_predicted.append(p.item())
			total_labels.append(l.item())
			if g not in group_total:
				group_total[g] = 0
				group_correct[g] = 0
			group_total[g] += 1
			per_digit_total[l.item()] += 1
			# if correct increase correct count
			if p.item() == l.item():
				per_digit_correct[l.item()] += 1
				group_correct[g] += 1
				true_positive[l.item()] += 1
			else:
				false_positive[p.item()] += 1
				false_negative[l.item()] += 1
		correct = (predicted == labels).sum().item()
		total = labels.size(0)

	return outputs, loss, correct, total, predicted, labels, per_digit_correct, per_digit_total, group_correct, group_total, total_predicted, total_labels, true_positive, false_positive, false_negative

def testMNIST(model, batch_len):
	testingData = datasets.MNIST('.', download=True, train=False, transform=transforms.Compose
							  ([
							  transforms.ToTensor(), 
							  transforms.Normalize((0.5,), (0.5,))
							  ]))
	testloader = torch.utils.data.DataLoader(testingData, 
										  batch_size=batch_len, 
										  shuffle=False, 
										  num_workers=2)
	per_digit_correct = [0] * 10
	per_digit_total = [0] * 10
	outputs, loss, total, correct = [], 0, 0, 0
	total_predicted = []
	total_labels = []
	true_positive = [0] * 10
	false_positive = [0] * 10
	false_negative = [0] * 10

	for images, labels in iter(testloader):
		fake_groups = torch.zeros_like(labels)
		outputsI, lossI, correctI, totalI, predicted, labelsI,_,_,_,_,curr_predicted,curr_labels,curr_true_positive, curr_false_positive, curr_false_negative = testImages(model, images, labels, fake_groups)

		for i in range(10):
			true_positive[i] += curr_true_positive[i]
			false_positive[i] += curr_false_positive[i]
			false_negative[i] += curr_false_negative[i]
		total_predicted += curr_predicted
		total_labels += curr_labels
		loss += lossI
		correct += correctI
		total += totalI

		# individual digit tracking for bar graphing later
		for i in range(len(predicted)):
			p = predicted[i]
			l = labelsI[i]
			per_digit_total[l.item()] += 1
			# if correct increase correct count
			if p.item() == l.item():
				per_digit_correct[l.item()] += 1

		for item in outputsI:
			outputs.append(item)
	outputs = torch.Tensor(np.array(outputs))
	return outputs, loss, correct, total, per_digit_correct, per_digit_total, true_positive, false_positive, false_negative
