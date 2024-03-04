# Installation

(optional) create a conda environment  
```
conda create -n myenv python=3.10
conda activate myenv
```

install requirements of the sample code: in the toplevel folder
```
pip install -r requirements.txt
```

install requirements of our code: 
```
pip install -r TeamCode/requirements.txt
```

# Run Tests
There are some relative paths in the tests. These assume you run them from the toplevel of this repo. 

In the toplevel run
```
    python -m pytest TeamCode
```

manually try the training loop as the challenge organizers do it for submissions:
```
python train_model.py -d TeamCode/tests/resources/example_data -m mymodel
python run_model.py -d TeamCode/tests/resources/example_data -m model -o test_outputs
```

# Developing
Only change code inside the `TeamCode` folder. 

Implement subclasses of `AbstractDigitizationModel` and `AbstractClassificationModel` specified in `interface.py`. 
Check out `test_sample_implementation` in `tests/test_end2end_models.py` to see how these are used. If you set up VSCode properly you should be able to step through the test with the debugger. Clemens can help. 

