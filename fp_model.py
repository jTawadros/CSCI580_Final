from torch import nn

IMAGE_IN = 784
IMAGE_OUT = 10

def initLayer(layer):
	layer.weight.data.normal_(std=0.01)
	layer.bias.data.fill_(0)

class NPlus1LenModel(nn.Module):
	def __init__(self, middle_alg, middle_sizes):
		super().__init__()
		self.middle_alg = middle_alg
		self.middle_length = len(middle_sizes)

		self.fcHide = nn.Linear(IMAGE_IN, middle_sizes[0])
		initLayer(self.fcHide)

		self.fcBetween = nn.ModuleList([])
		for i in range(1, self.middle_length):
			self.fcBetween.append(nn.Linear(middle_sizes[i-1], middle_sizes[i]))
			initLayer(self.fcBetween[i-1])

		self.fcOut = nn.Linear(middle_sizes[self.middle_length-1], IMAGE_OUT)
		initLayer(self.fcOut)

	def forward(self, data, labels):
		data = self.middle_alg(self.fcHide(data))
		for layer in self.fcBetween:
			data = self.middle_alg(layer(data))
		data = nn.LogSoftmax(dim=1)(self.fcOut(data))
		return data, nn.NLLLoss()(data, labels)

	def printParam(self):
		print("Hidden Layer 1 Weights --", self.fcHide.weight)
		print("Hidden Layer 1 Biases --", self.fcHide.bias)
		for i in range(1, self.middle_length):
			print("Hidden Layer", i+1, "Weights --", self.fcBetween[i-1].weight)
			print("Hidden Layer", i+1, "Biases --", self.fcBetween[i-1].bias)
		print("Output Layer Weights --", self.fcOut.weight)
		print("Output Layer Biases --", self.fcOut.bias)
