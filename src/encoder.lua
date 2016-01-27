require "torch"
require "nn"

patterns = torch.eye(8) * 2 - 1
targets = patterns

dataset = {};
function dataset:size() return 8 end
for i = 1, dataset:size() do
    dataset[i] = {patterns[i], patterns[i]}
end

-- classic 8-3-8 encoder
inputs = 8
hidden = 3
outputs = 8

mlp = nn.Sequential()
mlp:add(nn.Linear(inputs, hidden))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hidden, outputs))

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.1
trainer.maxIteration = 1500
trainer.verbose = true
trainer:train(dataset)

-- should have 8 unique columns exluding bias, sometimes weights are just
-- around the thresh and gives duplicates
print(torch.sign(mlp.modules[1].weight))
