require "torch"
require "gnuplot"

-- create linear seperable data
cols = 200
acols = cols / 2
bcols = cols / 2

a = torch.Tensor(2, acols)
a[1] = torch.randn(1, acols) * 0.5 + 1.0
a[2] = torch.randn(1, acols) * 0.5 + 0.5

b = torch.Tensor(2, bcols)
b[1] = torch.randn(1, bcols) * 0.5 - 1.0
b[2] = torch.randn(1, bcols) * 0.5

patterns = torch.Tensor(2, cols)
patterns[1][{{1, acols}}] = a[1]
patterns[2][{{1, acols}}] = a[2]
patterns[1][{{acols + 1, cols}}] = b[1]
patterns[2][{{acols + 1, cols}}] = b[2]

targets = torch.ones(cols, 1)
targets[{{acols + 1, cols}}] = -1

-- shuffle data
permute = torch.randperm(cols)
x = torch.Tensor(2, cols)
y = torch.Tensor(cols, 1)
for i=1, cols do
    x[1][i] = patterns[1][permute[i]]
    x[2][i] = patterns[2][permute[i]]
    y[i] = targets[permute[i]]
end

-- Delta update rule. It can be used for one training pattern at a time but
-- since all patterns should be summed we can do it with a matrix.
function delta(w, x, t, eta)
    return (w * x - t) * x:t() * -eta
end

-- Plot separation line from specified weights and with two clusters c1 and
-- c2.
function deltaplot(w, c1, c2)
    s = w:size()
    p = w[1][{{1, s[2] - 1}}]
    k = -w[1][s[2]] / (p * p)
    l = math.sqrt(p * p)

    from = torch.Tensor({p[1], p[1]}) * k + torch.Tensor({-p[2], p[2]}) / l
    to = torch.Tensor({p[2], p[2]}) * k + torch.Tensor({p[1], -p[1]}) / l

    gnuplot.grid(true)
    gnuplot.title("Delta Rule - Separation Line")
    gnuplot.plot({c1[1], c1[2], '+'}, {c2[1], c2[2], '+'}, {from, to, '-'})
end

-- Classify with weights w one input pattern. It will return 1 or -1.
function classify(w, x, y)
    return torch.sign(w * torch.Tensor({x, y, 1}))
end

inputs = 2
outputs = 1

W = torch.randn(1, inputs + 1)
X = torch.Tensor(inputs + 1, cols)
X[1] = x[1]
X[2] = x[2]
X[3] = torch.ones(1, cols) -- add bias
T = y

eta = 0.001
epochs = 40

for epoch = 1, epochs do
    W = W + delta(W, X, T, eta)
    deltaplot(W, a, b)
    os.execute("sleep 0.1") -- slow down for plot animation
end
