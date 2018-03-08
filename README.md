# Object-segmentation-defined-by-Query-expressions
Given an image and a natural language expression as query, the goal was to output a segmentation mask for the visual entities 
described by the expression. 
• As the project was for NLP class, we concentrated more on optimizing that part of the project. 
To accomplish this goal, we designed a model with three main components: 
• A Natural language expression encoder based on a recurrent LSTM network i.e. the most important part of our project, 
fully convolution network to extract local image descriptors and generate a spatial feature map
and a fully convolution classification and up-sampling network that takes as input the encoded expression and the spatial 
feature map and outputs a pixel-wise segmentation mask. 
• Language/technologies: Tensorflow, Python, Scikit-learn, NLTK, Matlab
