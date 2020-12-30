"""This module gathers the basic functions of a perceptron."""
module Perceptron

"""Mathematical function that determines the output of the perceptron.
   Activation function: Binary Step.
"""
function _activation_function(x::Float64)
    return x >= 0 ? 1 : 0
end

"""Try to find the best sets of weights and bias."""
function fit(
    X_train::Array{Float64,2},
    y_train::Array{Float64,1},
    epochs::Int64,
    β::Float64=-1.0,
    η::Float64=0.1
)

    number_samples = size(X_train,1)
    number_inputs = size(X_train,2)
    weights = rand(number_inputs)

    for epoch in 1:epochs
        no_errors = true

        for i in 1:number_samples
            u = sum(@.X_train[i,:]*weights) + β
            y = _activation_function(u)
            if y != y_train[i]
                error = y_train[i]-y
                β += η*error
                weights[:] = @.weights+η*error*X_train[i,:]
                no_errors = false
            end

        end

        if no_errors==true
            println("Trained in $epoch epochs.")
            break

        end

    end

    return weights, β
end

"""Predict output based on input, weights and bias."""
function predict(
    X::Array{Float64,1},
    weights::Array{Float64,1},
    β::Float64
)

    u=sum(@.X*weights) + β
    y=_activation_function(u)

    return y
end

end  # module Perceptron
