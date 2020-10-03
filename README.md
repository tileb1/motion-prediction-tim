# motion-prediction-tim

### Download data

The human3.6m dataset is in exponential map format.

```bash
git clone https://github.com/tileb1/motion-prediction-tim
cd motion-prediction-tim
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```
### Dependencies
* Download [PyTorch](https://pytorch.org/) with cuda (tested with 1.6.0)
* ```pip install -r requirements.txt```

### Train the model
Run the [main_3d.py](main_3d.py) file. Command line args are defined in [opt.py](utils/opt.py).
```bash
python main_3d.py
```

### Results
Average MPJE for short term prediction over 5 runs (all actions included)

| 80ms | 160ms | 320ms | 400ms |
|------|-------|-------|-------|
| 11.4 | 24.3  | 50.4  | 60.9  |

Average MPJE for long term prediction over 5 runs (walking, eating, discussion, smoking)

| 560ms | 1000ms |
|-------|--------|
| 49.6  | 68.6   |

Long term visualisation of our method compared to previous SOTA Mao et al. (2019)

![Alt Text](gif/visualisation.gif)

### Citing

If you use our code, please cite our work

```
@inproceedings{,
  title={},
  author={},
  booktitle={ACCV},
  year={2020}
}
```

### Acknowledgments

This code builds on top of [LearnTrajDep](https://github.com/wei-mao-2019/LearnTrajDep) by Mao et al. (2019).