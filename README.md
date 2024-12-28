# DBCG-Med:Diffusion-Based Bidirectional Calibration and Context Guidance for Medical Image Segmentation
## Experimental Environment
we use Ubuntu 18.04, Python 3.7, Pytorch 1.12.0.
## Abstract
Medical image segmentation faces inherent challenges, including irregular lesion shapes, complex backgrounds, and noise interference. Recently, diffusion models have achieved remarkable success in various generative tasks due to their powerful ability to model complex data distributions. To address these challenges, we propose a novel diffusion model segmentation framework tailored for medical images, called DBCG-Med. Our framework introduces a Biaxial Attention Module (BAM), which combines the orientation information of lesions by decomposing and weighting horizontal and vertical components to enhance spatial awareness. Furthermore, since diffusion models usually require long training time, we integrate Do-Conv into the feature encoder responsible for extracting features from the original image. This integration enhances feature extraction capabilities while improving convergence speed, training stability, and generalization capabilities. To provide consistent and stable features in the denoising stage, we design a dual context guided module (DCGM) to alleviate the encoder-decoder semantic gap. Experiments on four public medical image datasets show that our method achieves superior segmentation performance compared with mainstream methods, highlighting its great potential for practical application in clinical scenarios.
## Keywords
Diffusion model  Medical image segmentation  Bidirectional Adaptive Calibration  Global-Local Context Guidance
## Codes
We would upload our code here as soon as possible, please wait.
