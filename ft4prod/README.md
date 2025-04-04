# Ubuntu环境 

检查GPU是否可用 
nvidia-smi 

检查CUDA Toolkit是否可用 
nvcc –-version 

创建Python虚拟环境 

sudo apt update 
sudo apt install python3.10-venv 
 
python3 -m venv ~/jupyter_gpu 
 
source ~/jupyter_gpu/bin/activate 

安装Jupyter和GPU支持库 
 
# 更换 pypi 源加速库的安装 
 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
 
pip install --upgrade pip 
 
 
# 安装Jupyter核心组件 
 
pip install notebook jupyterlab ipykernel 
 
 
# 安装GPU支持框架（任选其一） 
 
# 选项1：TensorFlow 
 
pip install tensorflow[and-cuda] 
 
 
# 选项2：PyTorch 
 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
 
 
# 注册内核 
 
python -m ipykernel install --user --name=jupyter_gpu 
 

配置Jupyter Notebook 
# 生成配置文件 
 
jupyter notebook --generate-config 
 
 
# 设置密码 
 

jupyter notebook password 
 
 
# 编辑配置文件 
 

nano ~/.jupyter/jupyter_notebook_config.py 
 
添加以下内容： 
c.ServerApp.ip = '0.0.0.0' 
 
c.ServerApp.open_browser = False 
 
c.ServerApp.port = 8888 
 
c.ServerApp.allow_root = False 
 
c.ServerApp.allow_remote_access = True 
 

启动服务 
 
jupyter notebook 
 
如果是root登录的： 
jupyter notebook --allow-root 
 

验证GPU支持 
在Jupyter Notebook中创建新笔记本，运行： 
 
# TensorFlow验证 
import tensorflow as tf 
print("TF Version:", tf.__version__) 
print("GPU可用:", tf.config.list_physical_devices('GPU')) 
 
# PyTorch验证 
import torch 
print("PyTorch Version:", torch.__version__) 
print("CUDA可用:", torch.cuda.is_available()) 
print("设备数量:", torch.cuda.device_count()) 

 

self-llm/models/Qwen2.5/05-Qwen2.5-7B-Instruct Lora 微调.md at master · datawhalechina/self-llm · GitHub 
 
 

在notebook中最开始运行下面的命令安装所需要的库 
 
!pip install datasets 

!pip install tf-keras 

!pip install ipywidgets 

!pip install "jinja2>=3.1.0" 

!pip install modelscope==1.18.0 

!pip install transformers==4.44.2 

!pip install streamlit==1.24.0 

!pip install sentencepiece==0.2.0 

!pip install accelerate==0.34.2 

!pip install datasets==2.20.0 

!pip install peft==0.11.1 