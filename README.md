# UVMap-ID: A Controllable and Personalized UV Map Generative Model

Weijing Wang $^\star$, [Jichao Zhang](https://scholar.google.com/citations?user=SPEECTIAAAAJ&hl=en) $\star$ $^\dagger$, 
Chang Liu, [Xia Li](https://xialipku.github.io/), [Xingqian Xu](https://scholar.google.com/citations?user=s1X82zMAAAAJ&hl=en&oi=ao), 
[Humphrey Shi](https://www.humphreyshi.com/), [Nicu Sebe](http://disi.unitn.it/~sebe/), [Bruno Lepri](https://scholar.google.com/citations?user=JfcopG0AAAAJ&hl=en&oi=ao)<br>
$\star$: Equal Contribution, $\dagger$: Corresponding Author <br>

Abstract: Recently, diffusion models have made significant strides in synthesizing realistic 2D human images based on provided text prompts. Building upon this, researchers have extended 2D text-to-image diffusion models into the 3D domain for generating human textures (UV Maps). 
However, some important problems about UV Map Generative models are still not solved, i.e., how to generate personalized texture maps for any given face image, and how to define and evaluate the quality of these generated texture maps. 
To solve the above problems, we introduce a novel method, UVMap-ID, which is a controllable and personalized UV Map generative model. Unlike traditional large-scale training methods in 2D, we propose to fine-tune a pre-trained text-to-image diffusion model which is integrated with a face fusion module for achieving ID-driven customized generation. 
To support the finetuning strategy, we introduce a small-scale attribute-balanced training dataset, including high-quality textures with labeled text and Face ID. 
Additionally, we introduce some metrics to evaluate the multiple aspects of the textures. Finally, both quantitative and qualitative analyses demonstrate the effectiveness of our method in controllable and personalized UV Map generation. 



## [Paper](https://arxiv.org/abs/2404.14568) | [Video Youtube](https://www.youtube.com/watch?v=KCHUWPtBe9o)


 Dilireba            |         Guofucheng         |  Bengio 
:-------------------------:|:--------------------------:|:-------------------------:
![](./imgs/dilireba3.gif)  | ![](./imgs/guofucheng.gif) |  ![](./imgs/bengio.gif)

## Results

 CVers as ID            |             Celebrites as ID             
:-------------------------:|:----------------------------------------:|
<img src="./imgs/test1.png" width="400">  | <img src="./imgs/test3.png" width="400"> | 

## Rendering

 ID            |           UV Map           |  SMPL render 
:-------------------------:|:--------------------------:|:-------------------------:
<img src="./imgs/dilireba.png" width="400">  | ![](./imgs/dilireba2.png) |  ![](./imgs/dilireba3.gif)

### Citation

```
@article{wang2024uvmap,
  title={UVMap-ID: A Controllable and Personalized UV Map Generative Model},
  author={Wang, Weijie and Zhang, Jichao and Liu, Chang and Li, Xia and Xu, Xingqian and Shi, Humphrey and Sebe, Nicu and Lepri, Bruno},
  journal={ACM MM},
  year={2024}
}
```