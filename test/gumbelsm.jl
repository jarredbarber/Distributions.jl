using Distributions, Random, Test, LinearAlgebra, Turing

Random.seed!(34567)

g = GumbelSoftmax([0.1, 0.1, 0.7, 0.1], 0.5)

@test length(g) == 4

@test length(rand(g)) == 4

@test logpdf(g, [0.5, 0.5, 0.5, 0.5]) < -2.7

@test isnan(logpdf(g, [0.0,0.0,1.0,0.0]))




function test_model()
    @model demo(x) = begin
        τ = 0.5
        p ~ Dirichlet([2.0, 2.0])

        for k in 1:10
            x[k] ~ GumbelSoftmax(p, τ)
        end
    end

    x = zeros(10, 2)
    d = Categorical([0.45, 0.55])
    for k in 1:10
        x[k, rand(d)] = 1.0
    end
    s = SMC(100)
    chn = sample(demo(x), s)
    println(chn)
    true
end

@test test_model()
