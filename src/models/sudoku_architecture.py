from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

class Sudoku:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be "channels last"
        # channels last: The input shape to the model is in the form of (height, width, depth)
        model = Sequential()
        inputShape = (height, width, depth)

        # First set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (5, 5), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Second set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # First set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        # Second set of FC => RELU layers
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        # Softmax classifier
        # The number of nodes in the output layer is equal to the number of classes
        # The softmax classifier is used to output the probability of each class
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # Return the constructed network architecture
        return model