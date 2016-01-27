require "torch"
require "nn"
require "gnuplot"

-- create non-linear seperable data
cols = 200
acols = cols / 2
bcols = cols / 2

a = torch.Tensor(2, acols)
a[1][{{1, acols / 2}}] = torch.randn(1, acols / 2) * 0.2 - 1.0
a[1][{{acols / 2 + 1, acols}}] = torch.randn(1, acols / 2) * 0.2 + 1.0
a[2] = torch.randn(1, acols) * 0.2 + 0.3

b = torch.Tensor(2, bcols)
b[1] = torch.randn(1, bcols) * 0.3
b[2] = torch.randn(1, bcols) * 0.3 - 0.1

patterns = torch.Tensor(2, cols)
patterns[1][{{1, acols}}] = a[1]
patterns[2][{{1, acols}}] = a[2]
patterns[1][{{acols + 1, cols}}] = b[1]
patterns[2][{{acols + 1, cols}}] = b[2]

targets = torch.ones(cols, 1)
targets[{{acols + 1, cols}}] = -1

gnuplot.figure(1)
gnuplot.title("Non-linear Seperable Data Points")
gnuplot.grid(true)
gnuplot.plot({a[1], a[2], '+'}, {b[1], b[2], '+'})

-- create {input, output} pairs
dataset = {};
function dataset:size() return cols end
for i=1, dataset:size() do
    local input = torch.Tensor(2)
    local output = torch.Tensor(1)
    input[1] = patterns[1][i]
    input[2] = patterns[1][i]
    output[1] = targets[i]
    dataset[i] = {input, output}
end

inputs = 2
hidden = 100
outputs = 1

mlp = nn.Sequential()
mlp:add(nn.Linear(inputs, hidden))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hidden, outputs))

eta = 0.01
epochs = 40

err = torch.Tensor(epochs, 1)
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = eta
trainer.maxIteration = epochs
trainer.hookIteration = function (self, iteration, current_err)
    err[iteration] = current_err * 100
end
trainer:train(dataset)

gnuplot.figure(2)
gnuplot.title("Classification Error")
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Error (Percent)')
gnuplot.axis({0, epochs, 0, 100})
gnuplot.grid(true)
gnuplot.plot(err, '~')
