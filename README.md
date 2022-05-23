
## Res_SCA

This is the PyTorch implementation of the manuscript "Residual guided coordinate attention for selection channel aware image steganalysis". 

## Requirements:
CUDA (10.2)
cuDNN (7.4.1)
python (3.6.9)

## Use
"Proposed-SCA-CovNet.py" and "Proposed-SCA-J-YeNet.py" are the main program in spatial and JPEG domain, respectively. 

"CovNet_SRM_filters.py" contains the 30 fixed SRM filters and used in Proposed-SCA-CovNet.py 

"SRM_kernels.npy" contains the 30 basic SRM filters and used in Proposed-SCA-J-YeNet.py 


Example: 

If you want to detect S-UNIWARD steganography method at 0.4 bpp (on GPU #1), you can enter following command:

"python3 Proposed-SCA-CovNet.py -alg S-UNIWARD -rate 0.4 -g 1"


## Note
If you have any question, please contact me. (revere.wei@outlook.com)
