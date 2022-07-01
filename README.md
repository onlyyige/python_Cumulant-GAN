# python_Cumulant-GAN
["Cumulant GAN"](https://arxiv.org/abs/2006.06625)的复现源代码.

本文实现了一种新的损失函数来训练生成性对抗网络（GAN），新的损失函数基于产生累积量 GAN 的累积量生成函数（CGF）。根据导出的变分公式，相应的优化问题等价于 Rényi 散度最小化，Rényi 族包括 Kullback–Leibler 散度（KLD）、反向 KLD、Hellinger 距离和 χ2 散度。Wasserstein GAN 也是累积量 GAN 的成员。在稳定性方面，累积量 GAN 对于线性鉴别器、高斯分布和标准梯度下降上升算法具有纳什均衡的线性收敛性。最后，通过实验证明，改变其两个超参数值 β 和 γ，累积量 GAN 能够在大范围的发散和距离之间平滑值，以及提高了 WGAN 的训练水平，图像生成更具鲁棒性。


## 代码环境

- Python, NumPy, TensorFlow 2, SciPy, Matplotlib

## 模型

- `python cumgan_gmm8.py --epochs 10000 --disc_iters 5 --beta 0 --gamma 0 --iteration 0 --sess_name gmm8`: Toy dataset (8 Gaussians). 
- `python cumgan_swissroll.py --epochs 10000 --disc_iters 5 --beta 0 --gamma 0 --iteration 0 --sess_name gmm8`: Toy dataset (Swiss roll). 
- `python gan_toy.py`: Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll)
- `python gan_mnist.py`: MNIST

## 参考文献

Y. Pantazis, D. Paul, M. Fasoulakis, Y. Stylianou and M. A. Katsoulakis, "Cumulant GAN," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3161127.
