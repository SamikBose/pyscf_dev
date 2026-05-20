
# PySCF developmental repo to run enhanced sampling ab-initio molecular dynamics simulation
-----------------------------------------------

This is a project forked from the parent PySCF repo. The goal is to integrate this with weighted ensemble-based strategies implemented in wepy to run enhanced sampling aiMD simulations. This is currently an actively developing branch. Please contact bosesami@msu.edu or emeottna@msu.edu if you have any question.


# Installation

1. Make sure to be on a dev node, have the required modules loaded, and have the Conda environment activated.

```bash
ssh dev-amd24-h200
ml purge && ml load Miniforge3 OpenBLAS CUDA
conda activate wepy-dev
```

2. Clone the `pyscf_dev` repository and enter the directory.

```bash
git clone https://github.com/SamikBose/pyscf_dev.git
cd pyscf_dev
```

3. Build the PySCF package using the Conda build script.

```bash
./conda/build.sh
```

# Citing PySCF

Please cite the original PySCF paper:

[Recent developments in the PySCF program package](https://doi.org/10.1063/5.0006074),
Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan,
*J. Chem. Phys.*, **153**, 024109 (2020). doi:[10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
