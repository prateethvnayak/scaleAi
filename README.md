# ScaleAI-Challenge
A problem to detect circles and its parameters from a noisy image

#### Problem:
The problem is to architect and train a model which is able to output the parameters of the circle present inside of a given image under the presence of noise. The model should output a circle parameterized by (row, column, radius) which specifies the center coordinates of the circle in the image and the radius of the circle. 

#### Deliverables: (All 3 required)
- Trained model and working find_circle method
- The standard output of the model training in a file called training output.txt make sure that the training loss is visible in the output logs.
- The code used to define & train the model

#### Approach:

- The problem is broken down into two-stage detection using supervised learning and traditional computer vision algorothm.

- The stage 1 involves training a Convolutional-AutoEncoder network with noisy images as the input and the original image as the label. The loss function is a binary cross-entropy loss. 

- The noisy images are normalized prior to training by normalizing using the largest pixel value. Hence pixel values lie in [0, 1]

- The network has a total of ~70k parameters (~6Mb). There are three encoder conv layers and two decoder Conv layers. The final layer output is a pixel-wise sigmoid. 

- The second stage of the detection invovles using traditional Computer Vision algorithm - Canny Edge Detector and Hough Transform (in scikit-learn) for detecting the circles in the denoised image. 

- The Result obtained is 0.97 iou precision at AP=0.7 (result checked on 100 images)

#### Requirements:
- Tensorflow 2.0
- scikit-learn
- matplotlib
- Shapely
- numpy