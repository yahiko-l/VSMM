# VSMM
Paper: A Variational Selection Mechanism for Article Comment Generation (VSMM)

## Dataset
The Netease dataset access [address](https://drive.google.com/file/d/10oVzGGwNy3QhauwXOZIOaxOI2OEQ1cx3/view?usp=sharing). Please refer to the paper for detailed dataset information.

```
@article{LIU2024121263,
title = {A variational selection mechanism for article comment generation},
journal = {Expert Systems with Applications},
volume = {237},
pages = {121263},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.121263},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423017657},
author = {Jiamiao Liu and Pengsen Cheng and Jinqiao Dai and Jiayong Liu},
keywords = {Article comment generation, Variational selection mechanism, Recurrent neural network, Hierarchical encoding},
abstract = {Article comment generation is a new and challenging task in natural language processing, which has recently received much attention from researchers. In the article comment generation, there are apparent distinctions and different perspectives on the comments of a single article. However, current researches ignore the one-to-many relationship between articles and comments, resulting in a lack of diversity and coherence in generated comments. To solve this problem, a variational selection mechanism model (VSMM) is proposed in our research. In this model, we construct a Gaussian mixture prior network to capture a richer latent space and generate comments with more diversity and informativeness. At the same time, VSMM maps latent variables into different semantic spaces through the selection mechanism to capture one-to-many relationships. Then we introduce a discriminator to distinguish whether the selected content is consistent with the reference comment content, thus improving the coherence of the generated comments. In addition, a hierarchical encoder with attention is introduced in the VSMM model, which can effectively solve the problem of long document encoding. Furthermore, we propose a multi-category article comment dataset to align closely with practical applications. Experiments on three datasets demonstrate that VSMM outperforms existing state-of-the-art comment generation methods in terms of diversity in single and multiple comment generations. Moreover, VSMM can generate fluent, diverse, and coherent comments on multi-category and topic-rich datasets.}
}
```
