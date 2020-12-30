include("Perceptron.jl")

print("\n:::::::::: A classification  ::::::::::")
println("\n:::::::::: ----------------- ::::::::::\n")

X_train = [
            0.1  0.4  0.7;
            0.3  0.7  0.2;
            0.6  0.9  0.8;
            0.5  0.7  0.1;
          ]

y_train = [1.; 0; 0; 1]

X_test = copy(X_train)

epochs = 200

weights, β = Perceptron.fit(X_train, y_train, epochs)

println("\n\n********* Predict ***********\n")

for i in 1:size(X_test,1)
    y=Perceptron.predict(X_test[i,:], weights, β)
    println("$(X_test[i,:]) ===> $y")
end
