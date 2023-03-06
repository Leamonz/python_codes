import torch
import torch.nn as nn
from torchviz import make_dot, make_dot_from_trace
from DCGAN import DCGAN
import onnx

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DCGAN.Generator(100, 3, 64)
model.load_state_dict(torch.load('DCGAN/checkpoints/Generator.ckpt'))
model = model.eval().to(device)

x = torch.randn((1, 100, 1, 1)).to(device)
NetVis = make_dot(model(x), params=dict(list(model.named_parameters())))
NetVis.format = 'png'
NetVis.directory = 'Visualization'
#
# with torch.no_grad():
#     torch.onnx.export(
#         model,  # 要转换的模型
#         x,  # 模型的任意一个输入
#         'DCGAN_GEN.onnx',  # 导出的onnx文件名
#         opset_version=11,  # ONNX的算子集版本
#         input_names=['latent noise'],  # 输入Tensor的名称（自取）
#         output_names=['generated image']  # 输出Tensor的名称（自取）
#     )
#
# # 检测模型是否保存成功
# onnx_model = onnx.load('DCGAN_GEN.onnx')
# onnx.checker.check_model(onnx_model)
# # 打印计算图
# print(onnx.helper.printable_graph(onnx_model.graph))
