# Class-Orderings
Code from
"[On Class Orderings for Incremental Learning](https://arxiv.org/pdf/2007.02145.pdf)",
CL-ICML 2020

## Confusion Matrix Ordering (CMO)
The provided [jupiter notebook](./CIFAR100_class_ordering.ipynb) explains how
to load the model and the dataset for the case of CIFAR-100 on ResNet-32 from the
results provided by the [FACIL](https://github.com/mmasana/FACIL) framework
(see below). Then, it shows how to visualize the confusion matrices, how to permute
them with a greedy approach, and how we proposed to use simulated annealing for the
different class orderings.

## Obtaining the models with FACIL
We use FACIL to both train the models needed for calculating the class ordering,
and to evaluate on the different approaches.
It is the code for the survey paper:
_**Class-incremental learning: survey and performance evaluation**_
[[paper](https://arxiv.org/abs/2010.15277)] [[code](https://github.com/mmasana/FACIL)].
It provides a (hopefully!) helpful framework to develop new
methods for incremental learning and analyse existing ones.

### Installation and usage
1. Clone our github repository:

```
git clone https://github.com/mmasana/Class-Orderings.git
```

2. Clone the [FACIL](https://github.com/mmasana/FACIL) repository:

<details>

  For more details, check out the
[HOW TO](https://github.com/mmasana/FACIL/blob/master/README.md#how-to-use).
  
  ```
  git clone https://github.com/mmasana/FACIL.git
  ```

</details>

3. Run the provided [script](./script_facil.sh) or modify it to the required model
and dataset.


4. Modify the file locations and different options (such as the number of tasks, classes
per task, ...) of the jupiter notebook as needed.

## Extending FACIL
Once the required class orderings are computed from the simulated annealing of
the jupiter notebook we can include them in the [FACIL](https://github.com/mmasana/FACIL)
framework to evaluate them on different approaches. A new version of the dataset
(CIFAR-100 in this case) can be added to the `dataset_config` by following the
information from
[here](https://github.com/mmasana/FACIL/tree/master/src/datasets#main-usage).
It basically consists on creating an instance of the CIFAR-100 dataset but adding
the transformation of `class_order` with the corresponding list mapping.


## Citation
```
@inproceedings{masana2020class,
  title={On Class Orderings for Incremental Learning},
  author={Masana, Marc and Twardowski, Bart{\l}omiej and van de Weijer, Joost},
  booktitle={Continual Learning Workshop at International Conference on Machine Learning (CL-ICML)},
  year={2020}
}
```
