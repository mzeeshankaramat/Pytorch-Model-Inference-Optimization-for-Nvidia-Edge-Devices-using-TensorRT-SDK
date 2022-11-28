# Pytorch Model Inference Optimization using TensorRT SDK
In this repository, you will find code and scripts, with proper documentation required to on optimize Pytorch Models inference on NVIDIA edge devices.

TensorRT will optimizes the inference and it could be evaluated using some performance metrics like inference time, Frame per second(fps). Even though there is improvement in inference but there is a trade off with accuracy. This drop is due to precision tuning where we reduce the precision of parameters. Experimentally there isnt significant drop in Accuracy

Lets get started, Start your Nvidia device and install python
Make sure you have following packages installed in your device:
- TensorRT
- Pytorch (compatible version with TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html#trt_2)
- Pycuda:

  Set the environment variables by adding following lines in your .bashrc file:
  ```
    export PATH=/usr/local/cuda/bin:${PATH}            
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} 
  ```

  It tells the dynamic link loaders where to look for shared libraries.

 
Now if you have everything set up on your device lets dive deep further...
The model being optimized is resNet50. I have tried it on YOLO as well, trust me it works for YOLO as well ;).
You can use model of your choice, all you need to know is about the scripts you need to run for optimization.
 
Follow these steps:
 
 1. First we need a model trained on our dataset. We will be using CIFAR 10 dataset. I hope you are already familier with deeplearning stuff and know how to train a model. If not I got you covered. You can use the "TrainingModel.ipynb" notebook for training the model. You need to save the model’s state_dict (.pth) as this will be used further. Please note that the training takes lot of time and resources so you can any device having good GPU for training. I have used Google Colab for training. In the testing section you see the performance of model on test data in terms of time, fps and accuracy. 

 2. If you are also using some different device for training, download the preprocessed data and place it in **Preprocessed_data/** directory in your jetson device so that it can be used for inference. Also download the save model and place it inside **weights/** directory.

 3. Evaluate the performance of unoptimized model on Jetson device by running EvaluateModel.py script. Note down the results as we will be comparing them after inference optimization. 


### Inference Optimization:
 
  Now we have a trained model and we know its performance before optimization. Lets optimize the model and compare the results.
  
 4. Converting the Model into ONNX:

     Run the python script *ONNXConversion.py*. It will save the the converted ONNX model for our pytorch model.
   
 5. Building Deserailized Engine:

    To build the engine we will use ONNX parser. Run the python script BuildEngine.py 
   
 
 ### Optimized Inference
 
 6. Run the python script, OptimizedInference.py using terminal and compare the results in terms of accuracy, fps and time.
   


### References:
- https://developer.nvidia.com/tensorrt
- https://www.learnopencv.com/ 
- https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt_210/tensorrt-user-guide/index.html
