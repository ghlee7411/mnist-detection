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
2021-12-19 10:17:24.122601: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/10
2021-12-19 10:17:24.441305: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
625/625 [==============================] - 12s 19ms/step - loss: 0.3322 - binary_accuracy: 0.8988 - true_positives: 25.0000 - true_negatives: 287576.0000 - false_positives: 279.0000 - false_negatives: 32120.0000
Epoch 2/10
625/625 [==============================] - 12s 19ms/step - loss: 0.2995 - binary_accuracy: 0.9014 - true_positives: 903.0000 - true_negatives: 287539.0000 - false_positives: 316.0000 - false_negatives: 31242.0000
Epoch 3/10
625/625 [==============================] - 12s 19ms/step - loss: 0.1218 - binary_accuracy: 0.9552 - true_positives: 20687.0000 - true_negatives: 284983.0000 - false_positives: 2872.0000 - false_negatives: 11458.0000
Epoch 4/10
625/625 [==============================] - 12s 19ms/step - loss: 0.0459 - binary_accuracy: 0.9856 - true_positives: 29069.0000 - true_negatives: 286337.0000 - false_positives: 1518.0000 - false_negatives: 3076.0000
Epoch 5/10
625/625 [==============================] - 12s 19ms/step - loss: 0.0262 - binary_accuracy: 0.9919 - true_positives: 30526.0000 - true_negatives: 286881.0000 - false_positives: 974.0000 - false_negatives: 1619.0000
Epoch 6/10
625/625 [==============================] - 12s 19ms/step - loss: 0.0173 - binary_accuracy: 0.9947 - true_positives: 31122.0000 - true_negatives: 287172.0000 - false_positives: 683.0000 - false_negatives: 1023.0000
Epoch 7/10
625/625 [==============================] - 11s 18ms/step - loss: 0.0121 - binary_accuracy: 0.9962 - true_positives: 31426.0000 - true_negatives: 287344.0000 - false_positives: 511.0000 - false_negatives: 719.0000
Epoch 8/10
625/625 [==============================] - 11s 18ms/step - loss: 0.0088 - binary_accuracy: 0.9973 - true_positives: 31647.0000 - true_negatives: 287491.0000 - false_positives: 364.0000 - false_negatives: 498.0000
Epoch 9/10
625/625 [==============================] - 11s 18ms/step - loss: 0.0066 - binary_accuracy: 0.9979 - true_positives: 31767.0000 - true_negatives: 287563.0000 - false_positives: 292.0000 - false_negatives: 378.0000
Epoch 10/10
625/625 [==============================] - 12s 18ms/step - loss: 0.0049 - binary_accuracy: 0.9985 - true_positives: 31872.0000 - true_negatives: 287645.0000 - false_positives: 210.0000 - false_negatives: 273.0000
2021-12-19 10:19:20.807232: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
32/32 [==============================] - 1s 16ms/step - loss: 0.0279 - binary_accuracy: 0.9928 - true_positives: 1568.0000 - true_negatives: 14317.0000 - false_positives: 46.0000 - false_negatives: 69.0000
2021-12-19 10:19:21.457149: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.

Process finished with exit code 0
```