include("Perceptron.jl")

print("\n:::::::::: OR GATE  ::::::::::")
println("\n:::::::::: -------- ::::::::::\n")

X_train = [
            0  0.;
            0  1 ;
            1  0 ;
            1  1 ;
          ]

y_train= [0.;  1 ; 1  ; 1]

X_test = copy(X_train)

epochs = 50

weights, β = Perceptron.fit(X_train, y_train, epochs)

println("\n\n********* Predict ***********\n")

for i in 1:size(X_test,1)
    y=Perceptron.predict(X_test[i,:], weights, β)
    println("$(X_test[i,:]) ===> $y")
end
