 # <p align=center> [ECCV 2024] Restoring Images in Adverse Weather Conditions via Histogram Transformer</p>

<div align="center">
 
[![paper](https://img.shields.io/badge/Histoformer-paper-blue.svg)](https://export.arxiv.org/abs/2407.10172)
[![arXiv](https://img.shields.io/badge/Histoformer-arXiv-red.svg)](https://export.arxiv.org/abs/2407.10172)
![poster](https://img.shields.io/badge/Histoformer-poster-green.svg)
![video](https://img.shields.io/badge/Histoformer-video-orange.svg)
[![](https://img.shields.io/badge/Histoformer-supp-purple)](https://sunsean21.github.io/resources/eccv2024_supp.pdf)     
[![](https://img.shields.io/badge/chinese_blog-zhihu-blue.svg)]() 
[![Closed Issues](https://img.shields.io/github/issues-closed/sunshangquan/Histoformer)](https://github.com/sunshangquan/Histoformer/issues?q=is%3Aissue+is%3Aclosed) 
[![Open Issues](https://img.shields.io/github/issues/sunshangquan/Histoformer)](https://github.com/sunshangquan/Histoformer/issues) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsunshangquan%2FHistoformer&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
<div align=center>
<img src="assets/eccv2024_cover.jpg" width="720">
</div>

---
>**Restoring Images in Adverse Weather Conditions via Histogram Transformer**<br>  Shangquan Sun, Wenqi Ren, Xinwei Gao, Rui Wang, Xiaochun Cao<br> 
>European Conference on Computer Vision

> **Abstract:** *Transformer-based image restoration methods in adverse weather have achieved significant progress. Most of them use self-attention along the channel dimension or within spatially fixed-range blocks to
reduce computational load. However, such a compromise results in limitations in capturing long-range spatial features. Inspired by the observation that the weather-induced degradation factors mainly cause similar occlusion and brightness, in this work, we propose an efficient Histogram Transformer (Histoformer) for restoring images affected by adverse weather. It is powered by a mechanism dubbed histogram self-attention, which sorts and segments spatial features into intensity-based bins. Self-attention is then applied across bins or within each bin to selectively focus on spatial features of dynamic range and process similar degraded pixels of the long range together. To boost histogram self-attention, we present a dynamic-range convolution enabling conventional convolution to conduct operation over similar pixels rather than neighbor pixels. We also observe that the common pixel-wise losses neglect linear association and correlation between output and ground-truth. Thus, we propose to leverage the Pearson correlation coefficient as a loss function to enforce the recovered pixels following the identical order as ground-truth. Extensive experiments demonstrate the efficacy and superiority of our proposed method. We have released the codes in Github*
---

## News 🚀
* **2024.07.18**: Codes and pre-trained weights are released.
* **2024.07.17**: Visual results are released.
* **2024.07.14**: Arxiv Paper is released.
* **2024.07.01**: Histoformer is accepted by ECCV2024.

## Visual Results
| Method | RainDrop-a| Outdoor-Rain | Snow100K-L | Snow100K-S  | RealSnow |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|
| Restormer | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | 
| TransWeather | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | 
| WGWSNet | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) |
| Chen et al. | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) |  
| WeatherDiff | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | 
| Histoformer | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | [Results](<>) | 

In progress!