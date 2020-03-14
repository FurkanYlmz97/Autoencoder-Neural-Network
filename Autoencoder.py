import h5py as h5
import numpy as np
from matplotlib import pyplot as plt


def sigmoid_activation_function(input):
    return 1 / (1 + np.exp(-input))


def derivative_of_sigmoid(x):
    return sigmoid_activation_function(x) * (1 - sigmoid_activation_function(x))


def preprocessing(data):

    gray_data = np.zeros((10240, 16, 16))
    i = 0
    for images in data:
        gray_data[i] = 0.2126 * images[0] + 0.7152 * images[1] + 0.0722 * images[2]
        gray_data[i] = gray_data[i] - np.mean(gray_data[i])
        std_3 = np.std(gray_data[i]) * 3
        for m in range(16):
            for n in range(16):
                if gray_data[i][m][n] > std_3:
                    gray_data[i][m][n] = std_3
                elif gray_data[i][m][n] < -std_3:
                    gray_data[i][m][n] = -std_3

        gray_data[i] = gray_data[i] - np.min(gray_data[i])

        if np.max(gray_data[i]) == 0:
            gray_data[i] = gray_data[i] + 0.1
        else:
            gray_data[i] = gray_data[i] / np.max(gray_data[i]) * 0.8 + 0.1

        i += 1

    # Randomly plot 200 images both in RGB and Gray Scale
    for l in range(20):
        plt.figure()
        for p in range(10):

            # Randomly plot a RGB image
            rand = np.random.randint(0, 10239)
            plt.subplot(5, 4, 2*p + 1)
            plt.title("RGB: " + str(l * 10 + (p + 1)))
            a = np.swapaxes(data[rand], 0, 2)
            a = np.swapaxes(a, 0, 1)
            plt.axis("off")
            plt.imshow(a)

            # Plot the same image's Gray version
            plt.subplot(5, 4, 2*p + 2)
            plt.title("Gray: " + str(l * 10 + (p + 1)))
            plt.axis("off")
            plt.imshow(gray_data[rand], cmap='gray')
        plt.tight_layout()
    return gray_data


def training(data, beta, prob, epoch_number, Hidden_Neuron):

    # Specify parameters
    w0 = np.sqrt(6 / (256+Hidden_Neuron))
    lamda = 0.0005

    # Initialize weights
    w_1 = np.random.uniform(-w0, w0, (Hidden_Neuron, 256 + 1))
    w_2 = np.random.uniform(-w0, w0, (256, Hidden_Neuron + 1))

    # This part is taken from my previous assignment and than re-modified
    for m in range(epoch_number):

        x = data.reshape(10240, 256)  # input 10240x3x16x16
        d = x  # desired output
        x = np.hstack((x, np.atleast_2d(np.ones(10240)).T))

        # Hidden layer equations
        o_hidden = sigmoid_activation_function(np.matmul(x, w_1.transpose()))
        o_hidden = np.hstack((o_hidden, np.atleast_2d(np.ones(10240)).T))
        f_hidden = derivative_of_sigmoid(np.matmul(x, w_1.transpose()))

        # Output layer equations
        o = sigmoid_activation_function(np.matmul(o_hidden, w_2.transpose()))
        f = derivative_of_sigmoid(np.matmul(o_hidden, w_2.transpose()))

        # Calculating gradient of the output layer's weights
        delta_o = np.multiply(f, (d - o))
        gradient_w_2 = np.matmul(delta_o.transpose(), o_hidden)

        # Calculating gradient of the hidden layer's weights
        delta_1 = np.multiply(np.matmul(delta_o, w_2[0:, 0:Hidden_Neuron]), f_hidden)
        gradient_w_1 = np.matmul(delta_1.transpose(), x)

        # Update the weights according to gradients and reinitialize gradients
        pj = np.matmul(f_hidden.transpose(), x)
        ones = np.ones((Hidden_Neuron,257))
        KL_deriv = -prob * np.divide(ones, pj) + (1 - prob) * np.divide(ones, 1 - pj)
        w_1 = w_1 - 0.05 * (-gradient_w_1 / 256 + lamda * w_1 + beta * KL_deriv)
        w_2 = w_2 - 0.05 * (-gradient_w_2 / 256 + lamda * w_2)

    # Add bias
    x = data.reshape(10240, 256)  # input 10240x3x16x16
    x = np.hstack((x, np.atleast_2d(np.ones(10240)).T))

    # Hidden layer equations
    o_hidden = sigmoid_activation_function(np.matmul(x, w_1.transpose()))
    o_hidden = np.hstack((o_hidden, np.atleast_2d(np.ones(10240)).T))

    # Output layer equations
    o = sigmoid_activation_function(np.matmul(o_hidden, w_2.transpose()))

    # Plot the output and input together
    plt.figure()
    for p in range(10):

        rand = np.random.randint(0, 10239)
        plt.subplot(5, 4, 2*p + 1)
        plt.title("Input: " + str(p + 1))
        plt.axis("off")
        plt.imshow(data[rand], cmap='gray')

        plt.subplot(5, 4, 2*p + 2)
        plt.title("Output: " + str(p + 1))
        plt.axis("off")
        plt.imshow(o[rand].reshape((16, 16)), cmap='gray')
    plt.tight_layout()
    z = 0

    # Plot the weight images
    if Hidden_Neuron == 64:
        # For 64 hidden neuron
        for l in range(4):
            plt.figure()
            for p in range(8):

                plt.subplot(4, 4, 2*p + 1)
                plt.title("Weight: " + str(l * 16 + (2*p + 1)))
                plt.axis("off")
                plt.imshow(w_1[l * 16 + (2*p)][0:256].reshape(16, 16), cmap='gray')

                plt.subplot(4, 4, 2*p + 2)
                plt.title("Weight: " + str(l * 16 + (2*p + 2)))
                plt.axis("off")
                plt.imshow(w_1[l * 16 + (2*p + 1)][0:256].reshape(16, 16), cmap='gray')

            plt.tight_layout()

    elif Hidden_Neuron == 10:
        # For 10 Hidden Neuron
        plt.figure()
        for p in range(5):

            plt.subplot(3, 4, 2*p + 1)
            plt.title("Weight: " + str((2*p + 1)))
            plt.axis("off")
            plt.imshow(w_1[(2*p)][0:256].reshape(16, 16), cmap='gray')

            plt.subplot(3, 4, 2*p + 2)
            plt.title("Weight: " + str((2*p + 2)))
            plt.axis("off")
            plt.imshow(w_1[(2*p + 1)][0:256].reshape(16, 16), cmap='gray')

        plt.tight_layout()

    elif Hidden_Neuron == 100:
        # For 100 Hidden Neuron
        for l in range(5):
            plt.figure()
            for p in range(10):

                plt.subplot(4, 5, 2*p + 1)
                plt.title("Weight: " + str(l * 20 + (2*p + 1)))
                plt.axis("off")
                plt.imshow(w_1[l * 20 + (2*p)][0:256].reshape(16, 16), cmap='gray')

                plt.subplot(4, 5, 2*p + 2)
                plt.title("Weight: " + str(l * 20 + (2*p + 2)))
                plt.axis("off")
                plt.imshow(w_1[l * 20 + (2*p + 1)][0:256].reshape(16, 16), cmap='gray')

            plt.tight_layout()


if __name__ == '__main__':
    # Load the data
    file = h5.File("assign3_data1.h5", "r")
    data = np.array(file["data"])  # 10240x3x16x16

    # Preprocess the data
    gray_data = preprocessing(data)

    # Train the NN
    training(gray_data, 0.2, 0.5, 10000, 64)
    plt.show()