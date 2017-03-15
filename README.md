# FCN4SIN
Fully-convolutional networks for recognizing slab identification numbers.

This code was implemented based on Python and Tensorflow to solve an industrial problem: the recognition of slab identification numbers (SINs) in factory scenes that were collected from an actual steelworks.
We employed a fully-convolutional network (FCN) with decomvolution layers that was proposed in "Fully Convolutional Networks for Semantic Segmentation" (Evan Shelhamer, CVPR 2015).
Our FCN was based on the architecture of VGG19 with reduced number of 1x1 convolution filters for addressing the problem of insufficient GPU memory.



