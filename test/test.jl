include("../src/DILSNN.jl")

# DILS.jl
#########
plusOne = Primitive(1, 2, x->[x[1]+1, sin(x[1])],
                          x->[1 ,     cos(x[1])]) # produces sin, uselessly
square = Primitive(1, 1, x->[x[1]^2], x->[2*x[1]])
topLevel = Box(1, 1, [1=>2, 1=>1])

ExState = DILSState(Ex, 0.1)

"""         ------------------------
            |         __           |
            |    -|+1|__          -|-- OUTPUT₁
OUTPUT₁_ERR-|-                     |
            |                     -|--INPUT₁_ERR
INPUT₁    --|-           -|^2|-    |
            |                      |
            ------------------------
"""
Ex = DILS([topLevel, plusOne, square], [[2,3], Int[], Int[]])

# s_, res = stream!(Ex, [1.], 4.; n=400, α=1e-3, σ=0.01, verbose=false)
#p = plot(1:length(res), res)

# DILSNN.jl
###########

ReLu = Primitive(1, 1, x -> [max(0,x[1])], x->[x[1] > 0 ? 1 : 0])
Tanh = Primitive(1, 1, x -> [tanh(x[1])], x->[1 - tanh(x[1])^2])
Const1 = Primitive(0, 1, x -> [1], x->Float64[])

d = nn(1, Int[], Tanh)
s = DILSState(d, 0.05)
freeze = fixed_wires(s)


# doesn't learn well w/ nontrivial NN structure
# state, res = run_nn!(Int[2,2], [1.], 0.5; n=1000, α=1e-4, σ=0.2)


# pureTanh = DILS([Box(1, 1, [1=>1]), Tanh],[[2],Int[]])
# pstate, res = stream!(pureTanh, [1.], 0.5; n=100, α=1e-1) # learns well

# biasTanh = DILS([Box(1, 1, [1=>1,0=>1]), Tanh, Const1],[[2,3],Int[],Int[]])
# pstate, res = stream!(biasTanh, [1.], 0.5; n=100, α=1e-1) # learns well

# biasTanh = DILS([Box(1, 1, [1=>1,0=>1]), Tanh, Const1],[[2,3],Int[],Int[]])
# pstate, res = stream!(biasTanh, [1.], 0.5; n=100, α=1e-1) # learns well

#p = plot(1:length(res), res)

