import fp_benchmark
import fp_model
from torch import nn, optim

BATCH_SIZE = 96
EPOCHS = 50

model = fp_model.NPlus1LenModel(nn.LeakyReLU(0.1), [100])
optimizer = optim.SGD(model.parameters(), lr = 0.1)

mnist_correct, class_correct = fp_benchmark.run("Selected", BATCH_SIZE, model, optimizer, EPOCHS, True)
print(f"mnist_correct: {mnist_correct}, class_correct: {class_correct}")
print("MNIST Proportion Correct: " + str(mnist_correct * 100) + "%" + "\nClass Proportion Correct: " + str(class_correct * 100) + "%")