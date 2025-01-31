# Quantization

https://huggingface.co/docs/transformers/en/main_classes/quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn’t be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

https://medium.com/@rakeshrajpurohit/model-quantization-with-hugging-face-transformers-and-bitsandbytes-integration-b4c9983e8996

Quantization is a technique used to reduce the precision of numerical values in a model. Instead of using high-precision data types, such as 32-bit floating-point numbers, quantization represents values using lower-precision data types, such as 8-bit integers. This process significantly reduces memory usage and can speed up model execution while maintaining acceptable accuracy.

https://adnanwritess.medium.com/quantization-a47ada2fdd8f
Quantization is a compression technique that involves mapping high-precision values to lower-precision ones. For a large language model (LLM), this means modifying the precision of its weights and activations, making it less memory-intensive. While this process can impact the model’s capabilities, including its accuracy, it often presents a worthwhile trade-off depending on the use case. In many scenarios, it is possible to achieve comparable results with significantly lower precision. Quantization improves performance by reducing memory bandwidth requirements and increasing cache utilization.

Instead of using high-precision data types like 32-bit floating-point numbers, quantization represents values using lower-precision data types, such as 8-bit integers. This approach significantly reduces memory usage and can speed up model execution while maintaining acceptable accuracy.

LLMs are generally trained with full (float32) or half precision (float16) floating-point numbers. One float16 has 16 bits, which equals 2 bytes, so a one-billion parameter model trained on FP16 would require two gigabytes.

The quantization process involves representing the range of FP32 weight values in a lower precision format, such as FP16 or even INT4 (4-bit integers). A typical example is converting FP32 to INT8.

The overall impact on the quality of an LLM depends on the specific quantization technique used.

https://github.com/huggingface/transformers/issues/23970
... the bitsandbytes library only works on CUDA GPU.

https://en.wikipedia.org/wiki/CUDA
In computing, CUDA is a proprietary parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for accelerated general-purpose processing, an approach called general-purpose computing on GPUs. CUDA was created by Nvidia in 2006. When it was first introduced, the name was an acronym for Compute Unified Device Architecture, but Nvidia later dropped the common use of the acronym and now rarely expands it.

CUDA is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements for the execution of compute kernels. In addition to drivers and runtime kernels, the CUDA platform includes compilers, libraries and developer tools to help programmers accelerate their applications.
