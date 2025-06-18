This is the official implementation of SimpleUNet, a novel eaily scalable yet high-performance semantic segmentation model for medical images(Only for 2D at this moment). Without fantastic novel components, SimpleUNet is implmented Simply based on traditional U-Net by integrating interpretation-friendly feature selection, model width contraining(i.e., fixed model width for simplcity) and adaptive feature fusion. By doing so, the overall parameters can be greatly slashed while only 0.67 MB variant can achieve comparable or even better performance against the SOTA models such as TransUNet, UNetv2, Tiny-UNet and ESKNet, especially for breast lesion segmentation in Ultrasound and endoscopic polyps while showcasing promising performance on skin lesion segmentation.

Also, the developed SimpleUNet can be simply advanced by replacing the common convolution blocks with novel components. However, it might be impossible to exhuasitvely explore these componnets. Nevertheless, we validate the extendability of our SimpleUNet on basis of ESKNet, and many thanks to the great work of ESKNet,  which can be found at https://github.com/CGPxy/ESKNet. However, the original codes of ESKNet were implemented via Tensorflow. We , therefore, replicated them via Pytorch instead, on which we futher extend our SimpleUNet as SimpleESKNet, which showed more competitive performance regarding breast lesion and endoscopic polyps segmentation. 

If you found this work interesting or helpful, please cite:
@misc{yu2025simpleneedefficientaccurate,
      title={Simple is what you need for efficient and accurate medical image segmentation}, 
      author={Xiang Yu and Yayan Chen and Guannan He and Qing Zeng and Yue Qin and Meiling Liang and Dandan Luo and Yimei Liao and Zeyu Ren and Cheng Kang and Delong Yang and Bocheng Liang and Bin Pu and Ying Yuan and Shengli Li},
      year={2025},
      eprint={2506.13415},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.13415}, 
}
