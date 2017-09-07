# ESPCN-U-Net
U-Net architecture with ESPCN for semantic segmentation

How to run code:

Download dataset
1. Make h5 files of training and test sets using build_h5_voc_dataset function in h5_util.py
2. Place h5 files in the appropriate folders
3. Make directories for models, logs and samples
4. Choose architecture using the following flags in main.py:
  * deconv_name: type of upsampling procedure (deconv or sub_pixel_conv)
  * up_architecture: decoder architecture to use. There are 5 possible options: 1,2,3,4 and 5 (best performing model is option 3). If deconv is chosen for upsampling then the normal U-Net decoder architecture is used.
  * activation_function: choose activation function to use (ReLU or parametric ReLU)
5. To train run the following command in terminal: python3 main.py
6. To test run the following command in terminal: python3 main.py --action test 
7. Use data_analysis.py file in utils folder to obtain metrics and confusion matrix.
8. To predict run the following command in terminal: python3 main.py --action predict (images will be saved in samples directory) 
