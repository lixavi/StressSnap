# StressSnap

The objective of this project was to detect human emotions from facial expressions using two datasets, FBE 2013 and CKPlus. The dataset consists of seven classes: angry, disgust, fear, happy, neutral, sad, and surprise, with each image size being 48x48 pixels. The project aimed to preprocess the data, fine-tune pre-trained models, and develop a final model with high accuracy in emotion detection which will be used to detect stress levels.

### ResNet + Attention
The ResNet with Attention models consists of three main components: the ResNet152 backbone, an attention module, and a fully connected layer. The ResNet152 backbone is pre-trained on a large dataset, which allows it to learn a rich set of features that can be used for a variety of computer vision tasks. The attention module is a set of convolutional layers that are used to compute a weight map that indicates which parts of the input image are most important for the classification task. The attention module is designed to learn which parts of the image contain the most discriminative information for the classification task.

The fully connected layer takes the output of the ResNet backbone and the attention module and produces a final output that represents the predicted class probabilities. The output of the fully connected layer is a vector of length 7, which corresponds to the number of classes that the model can predict.

During inference, the input image is first passed through the ResNet backbone to extract a set of features. The attention module is then used to compute a weight map that indicates which parts of the image are most important for the classification task. The features are then multiplied by the attention map, and the resulting features are passed through a global average pooling layer to produce a feature vector. The feature vector is then passed through the fully connected layer to produce the final output.

The ResNet with Attention model is trained using a standard cross-entropy loss function and the Adam optimizer. During training, the weights of the ResNet backbone and the attention module are frozen, and only the weights of the fully connected layer are updated. The model is fine-tuned on a specific dataset to learn to classify images into 7 classes.

### ResNet
The ResNet models have a modified fully connected layer that consists of two linear layers with ReLU activation in between. The first linear layer has an output of 128 neurons, and the second linear layer has an output of 7 neurons, which is the number of classes in the classification task. The purpose of the modified fully connected layer is to learn a more complex mapping between the output of the ResNet models and the target labels.

### VGG
Similarly, the last layer of VGG-16 was modified for our task and was fine-tuned.

### Results
| Accuracy | Loss |
| -------- | -------- |
| <img src="https://github.com/abuba8/StressSnap/tree/main/results/acc.JPG" alt="Accuracy plot of ResNetAttention Model"> | <img src="https://github.com/abuba8/StressSnap/tree/main/results/loss.JPG" alt="Training/Validation loss plot of ResNetAttention Model"> |

