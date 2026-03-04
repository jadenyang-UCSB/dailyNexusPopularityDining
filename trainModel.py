from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import losses, metrics
from tensorflow.keras.optimizers import Adam


def train(arrayPic, arrayCode):
    # print(arrayPic.shape)
    # print(arrayPic.dtype)
    # print(arrayCode.shape)
    # print(arrayCode.dtype)
    X_train, X_test, y_train, y_test = train_test_split(
    arrayPic, 
    arrayCode, 
    test_size = 0.25, 
    random_state = 42
    )

    convolveNN = Sequential([
        Conv2D(32,(3,3),activation = "relu", input_shape = (128,128,3)),
        MaxPooling2D(pool_size = (2,2)),
        Flatten(),
        Dense(64,activation = "relu"),
        Dense(1)
    ])


    #How come when I look at the libraries I can't see the optimzers, loss, and metrics as a parameter. What does mse stand for and why do I have to put paranthese around mae.
    convolveNN.compile(
        optimizer = Adam(learning_rate = 0.00001),
        loss = losses.MeanSquaredError(),
        metrics = [metrics.MeanAbsoluteError()]
    )
    #What are epochs and batch sizes
    testing = convolveNN.fit(
        X_train,y_train, 
        validation_data = (X_test,y_test),
        epochs = 10,
        batch_size = 32
    )

    convolveNN.save("peopleCounter.keras")
    
    return convolveNN, testing
    
