def test(X_test,Y_test,model):

    loss_and_metrics=model.evaluate(X_test, Y_test, batch_size=128)
    
    print(loss_and_metrics)

