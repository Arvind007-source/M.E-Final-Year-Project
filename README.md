# M.E-Final-Year-Project
DATASET
Name - KITTI Dataset
Description -
	A popular benchmark dataset for testing and developing computer vision algorithms, particularly in the context of robotics and autonomous 
	driving, is the KITTI dataset. The dataset was collected at the Karlsruhe Institute of Technology (KIT) and offers a comprehensive compilation 
	of real-world driving scenarios captured by high-resolution stereo cameras, 3D laser scanners, and GPS/IMU sensors. 
	A wide range of scenes that are necessary for training and testing algorithms in various aspects of autonomous driving are provided by the 
	KITTI dataset. It includes a variety of driving conditions, such as highway, country, and urban settings. The dataset is broken down into 
	several smaller sets, each focusing on a different job, such as optical flow, object detection, 3D object tracking, stereo vision, and so forth.
	The creation and assessment of depth estimate methods is made possible by the stereo vision subset, which offers stereo picture pairings 
	with matching disparity maps. The ground truth optical flow data annotated image sequences found in the optical flow subset are helpful 
	for training algorithms that estimate object motion between frames. since of its real world data, autonomous driving systems are made more 
	robust and reliable since models trained on KITTI can effectively generalize to real-world driving conditions. The dataset offers an authentic 
	and demanding environment for assessing and enhancing the performance of sophisticated algorithms, which makes it a valuable contribution to 
	the fields of computer vision and autonomous cars. The subsets for object identification and tracking comprise of photos that have been labeled 
	with bounding boxes and item labels. This makes it easier to train algorithms capable of identifying and tracking things in dynamic contexts, 
	such as cars and pedestrians. Because of its wide range of driving scenarios and rich annotations, the KITTI dataset is an invaluable tool for 
	developing autonomous driving technology. The KITTI dataset was captured while traveling through Karlsruhe, Germany, from a movable platform. 
	It consists of IMU accelerations from a combined GPS/IMU system, high-precision GPS measurements, camera pictures, and laser scans. 
	This dataset's primary goal is to advance the development of robotic and computer vision systems for autonomous driving. We provide 
	technical details on the raw data itself here to supplement the information in our introduction paper, which mostly focused on benchmarks, 
	their development, and their application for assessing cutting-edge computer vision techniques. We comment on sensor limits and potential 
	mistakes and provide detailed instructions on how to obtain the data. A calibrated, synced, and rectified autonomous driving dataset that 
	captures a variety of intriguing scenarios is described in this work. This dataset will be very helpful in many robotics and computer vision 
	applications. The intention to increase the number of sequences that are available in the future by labeling more 3D objects for sequences 
	that aren't yet tagged and by recording new sequences in challenging lighting conditions such at night, in tunnels, or in the rain or fog. 
	We also intend to add new challenges to our suite of benchmarks. Specifically, offering semantic labels that are correct to the pixel for 
	large number of the sequences.

DATASET URL - https://www.kaggle.com/datasets/klemenko/kitti-dataset

SOFTWARE AND HARDWARE REQUIREMENTS
SOFTWARE REQUIREMENTS
In order to properly run and test the given source code for this project, certain software components and libraries have to be installed 
and configured correctly in the development environment. Following is the operating system, Python version, and set of basic Python 
packages/libraries needed for supporting data processing, machine learning model training, and visualization. Below Listed are software Requirements 
for this Project:- 
+----------------------+--------------------------------------------------------+
| Component           | Requirements                                            |
+---------------------+---------------------------------------------------------+
| Operating System    | Windows 10/11, Ubuntu 20.04+, or macOS                  |
+---------------------+---------------------------------------------------------+
| Python Version      | 3.8 or above                                            |
+---------------------+---------------------------------------------------------+
| Libraries/Packages  | numpy                                                   |
|                     | pandas                                                  |
|                     | matplotlib                                              |
|                     | tensorflow / pytorch                                    |
|                     | scikit-learn                                            |
|                     | opencv-python                                           |
|                     | seaborn                                                 |
|                     | jupyter                                                 |
+---------------------+---------------------------------------------------------+

HARDWARE REQUIREMENTS
As shown in below table, the synthetic data generation ecosystem includes components such as simulation environments, scenario 
generators, and GAN models.

	Ecosystem for Synthetic Data Generation in Autonomous Vehicles

+-----------------------------------------------------------------------------------------+
| Component                | Description                                                  |
|-----------------------------------------------------------------------------------------|
|         			     HARDWARE                                                         |
|-----------------------------------------------------------------------------------------|                                                              
| GPUs                     | Multi-GPU Setup (e.g., NVIDIA A100)                          |
| Storage                  | High-Speed SSD (1TB+)                                        |
| Memory                   | 32GB+ RAM                                                    |
| Computing Infrastructure | Multi-GPU Setup, TensorFlow, PyTorch                         |
|                          |                                                              |
|-----------------------------------------------------------------------------------------|                                                                                       
|                   GENERATIVE AI MODELS                                                  |
|-----------------------------------------------------------------------------------------|                                                              
| Models Used              | GAN, VAE, StyleGAN, CGAN, Beta-VAE                           |
|-----------------------------------------------------------------------------------------|
|                LARGE LANGUAGE MODELS (LLMS)                                             |
|-----------------------------------------------------------------------------------------|         
| Model Used               | GPT-2 for Scenario Description                               |
|-----------------------------------------------------------------------------------------|
|                       DATASETS                                                          | 
|-----------------------------------------------------------------------------------------|                                                            
| Primary Dataset          | KITTI Dataset (Left/Right Images)                            |
| Augmented Data           | Custom Synthetic Data from VAE/GAN                           |
|                                                                                         |
|-----------------------------------------------------------------------------------------|
|                  EVALUATION METRICS                                                     |                                                              
| Metrics Used             | FID, ROUGE, Perplexity, Semantic Similarity                  |
|-----------------------------------------------------------------------------------------|
|                     APPLICATIONS                                                        | 
|-----------------------------------------------------------------------------------------|                                                             
| Use Cases                | AV Training, Data Augmentation, Safety Testing               |
+-----------------------------------------------------------------------------------------+

DETAILED STEPS TO EXECUTE THE CODE

i)  The Link to the dataset is:- https://github.com/Arvind007-source/M.E-Final-Year-Project.git
ii) Clone the Dataset, either in Kaggle or in Colab, or in Jupyter Notebook.
iii) Perform the Following Steps:-
    a) Open the Kaggle/ Colab/ Jupyter Notebook, and sign in to the Kaggle using your Google Account.
    b) Clone the Dataset.
    c) if using Kaggle Notebook, then click on Create button and click on New Notebook
        - After creating the new notebook,  Clone the Dataset
        - Then Turn on Settings -> Accelerator -> Select the GPU  

