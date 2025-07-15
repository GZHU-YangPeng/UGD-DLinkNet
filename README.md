# UGD-DLinkNet

This is the code of "UGD-DLinkNet: An Enhanced Network for Occluded Road Extraction Using Attention Mechanisms and Uncertainty Estimation".
The code will be released soon.

## Requirements：
```
PyTorch >= 2.0.0
Python  3.8
CUDA  11.8
```

### Train

**To train** model in different settings (locations, pairwise functions), please refer [here](https://github.com/yswang0522/NLLinkNetRoadExtraction/blob/master/run_example.sh).

To train **UGD-DLinkNet**(Massachusetts road dataset):

    python train_mass.py 

To train **UGD-DLinkNet**(CHN6-CUG Road Dataset)

    python train_chn6.py 

To train **UGD-DLinkNet**  (DeepGlobe dataset ) 

    python train_deep.py 

### Predict

**UGD-DLinkNet**(Massachusetts road dataset):

    python test_mass.py 

**UGD-DLinkNet**(CHN6-CUG Road Dataset)

    python test_chn6.py 

**UGD-DLinkNet**(DeepGlobe dataset ) 

    python test_deep.py 
