# MNIST-Detection
```commandline
Metal device set to: Apple M1 Max

systemMemory: 32.00 GB
maxCacheSize: 10.67 GB

Model: "MNIST-Detector"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 112, 112, 1  0           []                               
                                )]                                                                
                                                                                                  
 conv2d (Conv2D)                (None, 112, 112, 32  544         ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 conv2d_1 (Conv2D)              (None, 56, 56, 32)   16416       ['conv2d[0][0]']                 
                                                                                                  
 input_2 (InputLayer)           [(None, 10)]         0           []                               
                                                                                                  
 conv2d_2 (Conv2D)              (None, 28, 28, 32)   16416       ['conv2d_1[0][0]']               
                                                                                                  
 dense (Dense)                  (None, 6272)         68992       ['input_2[0][0]']                
                                                                                                  
 conv2d_3 (Conv2D)              (None, 14, 14, 32)   16416       ['conv2d_2[0][0]']               
                                                                                                  
 reshape (Reshape)              (None, 14, 14, 32)   0           ['dense[0][0]']                  
                                                                                                  
 concatenate (Concatenate)      (None, 14, 14, 64)   0           ['conv2d_3[0][0]',               
                                                                  'reshape[0][0]']                
                                                                                                  
 conv2d_4 (Conv2D)              (None, 7, 7, 32)     32800       ['concatenate[0][0]']            
                                                                                                  
 conv2d_5 (Conv2D)              (None, 4, 4, 1)      513         ['conv2d_4[0][0]']               
                                                                                                  
 reshape_1 (Reshape)            (None, 4, 4)         0           ['conv2d_5[0][0]']               
                                                                                                  
==================================================================================================
Total params: 152,097
Trainable params: 152,097
Non-trainable params: 0
__________________________________________________________________________________________________
2021-12-19 09:46:00.600306: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/20
2021-12-19 09:46:00.858129: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
625/625 [==============================] - 11s 16ms/step - loss: 0.3304 - accuracy: 0.2594 - precision: 0.0991 - recall: 0.0018
Epoch 2/20
625/625 [==============================] - 10s 16ms/step - loss: 0.2894 - accuracy: 0.3154 - precision: 0.8601 - recall: 0.0519
Epoch 3/20
625/625 [==============================] - 10s 16ms/step - loss: 0.1002 - accuracy: 0.4619 - precision: 0.9055 - recall: 0.7028
Epoch 4/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0391 - accuracy: 0.4862 - precision: 0.9550 - recall: 0.9211
Epoch 5/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0239 - accuracy: 0.4917 - precision: 0.9717 - recall: 0.9546
Epoch 6/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0164 - accuracy: 0.4956 - precision: 0.9793 - recall: 0.9677
Epoch 7/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0118 - accuracy: 0.5000 - precision: 0.9841 - recall: 0.9782
Epoch 8/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0088 - accuracy: 0.5000 - precision: 0.9885 - recall: 0.9827
Epoch 9/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0074 - accuracy: 0.4969 - precision: 0.9894 - recall: 0.9866
Epoch 10/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0056 - accuracy: 0.5020 - precision: 0.9922 - recall: 0.9899
Epoch 11/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0046 - accuracy: 0.5034 - precision: 0.9931 - recall: 0.9919
Epoch 12/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0044 - accuracy: 0.5005 - precision: 0.9928 - recall: 0.9917
Epoch 13/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0034 - accuracy: 0.5044 - precision: 0.9949 - recall: 0.9940
Epoch 14/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0031 - accuracy: 0.5060 - precision: 0.9953 - recall: 0.9946
Epoch 15/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0038 - accuracy: 0.5015 - precision: 0.9935 - recall: 0.9926
Epoch 16/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0028 - accuracy: 0.4984 - precision: 0.9953 - recall: 0.9949
Epoch 17/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0026 - accuracy: 0.5043 - precision: 0.9957 - recall: 0.9950
Epoch 18/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0027 - accuracy: 0.5006 - precision: 0.9957 - recall: 0.9951
Epoch 19/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0029 - accuracy: 0.5046 - precision: 0.9950 - recall: 0.9944
Epoch 20/20
625/625 [==============================] - 10s 16ms/step - loss: 0.0026 - accuracy: 0.5057 - precision: 0.9960 - recall: 0.9948
2021-12-19 09:49:21.336372: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
32/32 [==============================] - 1s 12ms/step - loss: 0.0224 - accuracy: 0.4838 - precision: 0.9786 - recall: 0.9721
2021-12-19 09:49:21.846841: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.

Process finished with exit code 0
```