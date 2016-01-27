require "torch"
require "nn"
require "gnuplot"

function cgauss(x)
    return torch.exp(torch.cmul(x, x) * -0.1)
end

function gaussplot(fig, z, title)
    gnuplot.figure(fig)
    gnuplot.raw("set key off")
    gnuplot.raw("set xrange [0:11]")
    gnuplot.raw("set yrange [0:11]")
    gnuplot.raw("set zrange [-0.9:0.9]")
    gnuplot.title(title)
    gnuplot.xlabel("x")
    gnuplot.ylabel("y")
    gnuplot.zlabel("z")
    gnuplot.splot(z)
end

x = torch.range(-5, 5)
y = x:clone()
z = torch.ger(cgauss(x), cgauss(y)) - 0.5

gaussplot(1, z, "Gauss function (Exact)")

xs = x:size()[1]
ys = y:size()[1]
ns = xs * ys -- grid size

xx = torch.Tensor(xs, xs)
yy = torch.Tensor(ys, ys)
for i = 1, xs do xx[i] = x:clone() end
for i = 1, ys do yy[i] = y:clone() end
yy = yy:t()

patterns = torch.Tensor(2, ns)
patterns[1] = torch.reshape(xx, 1, ns)
patterns[2] = torch.reshape(yy, 1, ns)
patterns = patterns:t()
targets = torch.reshape(z, ns, 1)

inputs = 2
hidden = 10 -- lower or higher with risk for under/overfit
outputs = 1

mlp = nn.Sequential()
mlp:add(nn.Linear(inputs, hidden))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hidden, outputs))

eta = 0.2
epochs = 350
criterion = nn.MSECriterion()

shuffle = torch.randperm(ns)
for epoch = 1, epochs do
    -- update weights
    criterion:forward(mlp:forward(patterns), targets)
    mlp:zeroGradParameters()
    mlp:backward(patterns, criterion:backward(mlp.output, targets))
    mlp:updateParameters(eta)
    -- plot progress
    zz = torch.reshape(mlp.output, xs, ys)
    gaussplot(2, zz, "Gauss Function (Approx) with " .. hidden .. " hidden nodes at epoch " .. epoch)
end
