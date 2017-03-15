# FCN4SIN
Fully-convolutional networks for recognizing slab identification numbers.

This code was implemented to solve an actual industry problem: the recognition of slab identification numbers (SINs) in factory scenes.
The images were collected from an actual steelworks, and this algorithm was implemented based on Python and Tensorflow.
The architecture of VGG19 was utilized to construct a fully-convolutional network, 
and reduced number of 1x1 convolution filters were used for addressing the problem of insufficient GPU memory.



