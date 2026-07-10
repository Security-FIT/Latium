# Latium Qwen3-8B Cluster Workflow

## Cluster access
- Jump host: 
- Target GPU host:  (NVIDIA A100 80GB)
- SSH chain: 

## Environment setup
branch 'refactor-clean' set up to track 'origin/refactor-clean'.
Requirement already satisfied: pip in /home/metju/miniconda3/lib/python3.13/site-packages (25.3)
Collecting pip
  Downloading pip-26.1.2-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: setuptools in /home/metju/miniconda3/lib/python3.13/site-packages (80.9.0)
Collecting setuptools
  Downloading setuptools-82.0.1-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: wheel in /home/metju/miniconda3/lib/python3.13/site-packages (0.45.1)
Collecting wheel
  Downloading wheel-0.47.0-py3-none-any.whl.metadata (2.3 kB)
Requirement already satisfied: packaging>=24.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from wheel) (25.0)
Downloading pip-26.1.2-py3-none-any.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 30.5 MB/s  0:00:00
Downloading setuptools-82.0.1-py3-none-any.whl (1.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 25.6 MB/s  0:00:00
Downloading wheel-0.47.0-py3-none-any.whl (32 kB)
Installing collected packages: wheel, setuptools, pip
  Attempting uninstall: wheel
    Found existing installation: wheel 0.45.1
    Uninstalling wheel-0.45.1:
      Successfully uninstalled wheel-0.45.1
  Attempting uninstall: setuptools
    Found existing installation: setuptools 80.9.0
    Uninstalling setuptools-80.9.0:
      Successfully uninstalled setuptools-80.9.0
  Attempting uninstall: pip
    Found existing installation: pip 25.3
    Uninstalling pip-25.3:
      Successfully uninstalled pip-25.3

Successfully installed pip-26.1.2 setuptools-82.0.1 wheel-0.47.0
Looking in indexes: https://download.pytorch.org/whl/cu121
Requirement already satisfied: torch in /home/metju/miniconda3/lib/python3.13/site-packages (2.11.0.dev20260128+cu128)
Requirement already satisfied: torchvision in /home/metju/miniconda3/lib/python3.13/site-packages (0.25.0.dev20260128+cu128)
Requirement already satisfied: torchaudio in /home/metju/miniconda3/lib/python3.13/site-packages (2.11.0.dev20260128+cu128)
Requirement already satisfied: filelock in /home/metju/.local/lib/python3.13/site-packages (from torch) (3.25.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (4.15.0)
Requirement already satisfied: setuptools in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (82.0.1)
Requirement already satisfied: sympy>=1.13.3 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (3.6.1)
Requirement already satisfied: jinja2 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (2025.10.0)
Requirement already satisfied: cuda-bindings==12.9.4 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.9.4)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.17.1.4 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (9.17.1.4)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.28.9 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (2.28.9)
Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (3.4.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (1.13.1.3)
Requirement already satisfied: triton==3.6.0+git9844da95 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch) (3.6.0+git9844da95)
Requirement already satisfied: cuda-pathfinder~=1.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from cuda-bindings==12.9.4->torch) (1.3.3)
Requirement already satisfied: numpy in /home/metju/miniconda3/lib/python3.13/site-packages (from torchvision) (2.3.5)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from torchvision) (12.1.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from jinja2->torch) (3.0.3)
Requirement already satisfied: transformers in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 1)) (5.0.0)
Requirement already satisfied: hydra-core in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 2)) (1.3.2)
Requirement already satisfied: hydra_colorlog in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 3)) (1.2.0)
Requirement already satisfied: omegaconf in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 4)) (2.3.0)
Requirement already satisfied: numpy in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 5)) (2.3.5)
Requirement already satisfied: scipy in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 6)) (1.17.0)
Requirement already satisfied: pandas in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 7)) (3.0.0)
Requirement already satisfied: seaborn in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 8)) (0.13.2)
Requirement already satisfied: matplotlib in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 9)) (3.10.8)
Requirement already satisfied: datasets in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 10)) (4.5.0)
Requirement already satisfied: accelerate in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 11)) (1.12.0)
Requirement already satisfied: tensorboard in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 12)) (2.20.0)
Requirement already satisfied: tqdm in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 13)) (4.67.1)
Requirement already satisfied: scikit-learn in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 14)) (1.8.0)
Requirement already satisfied: nltk in /home/metju/miniconda3/lib/python3.13/site-packages (from -r /dev/fd/63 (line 15)) (3.9.2)
Collecting ruff (from -r /dev/fd/63 (line 16))
  Downloading ruff-0.15.20-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (26 kB)
Collecting pytest (from -r /dev/fd/63 (line 17))
  Downloading pytest-9.1.1-py3-none-any.whl.metadata (7.6 kB)
Requirement already satisfied: filelock in /home/metju/.local/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (3.25.0)
Requirement already satisfied: huggingface-hub<2.0,>=1.3.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (1.3.4)
Requirement already satisfied: packaging>=20.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (6.0.3)
Requirement already satisfied: regex!=2019.12.17 in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (2026.1.15)
Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (0.22.2)
Requirement already satisfied: typer-slim in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (0.20.0)
Requirement already satisfied: safetensors>=0.4.3 in /home/metju/miniconda3/lib/python3.13/site-packages (from transformers->-r /dev/fd/63 (line 1)) (0.7.0)
Requirement already satisfied: fsspec>=2023.5.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (2025.10.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (1.2.0)
Requirement already satisfied: httpx<1,>=0.23.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (0.28.1)
Requirement already satisfied: shellingham in /home/metju/miniconda3/lib/python3.13/site-packages (from huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (1.5.4)
Requirement already satisfied: typing-extensions>=4.1.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (4.15.0)
Requirement already satisfied: anyio in /home/metju/miniconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (4.12.1)
Requirement already satisfied: certifi in /home/metju/miniconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (2025.11.12)
Requirement already satisfied: httpcore==1.* in /home/metju/miniconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (1.0.9)
Requirement already satisfied: idna in /home/metju/miniconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (3.11)
Requirement already satisfied: h11>=0.16 in /home/metju/miniconda3/lib/python3.13/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers->-r /dev/fd/63 (line 1)) (0.16.0)
Requirement already satisfied: antlr4-python3-runtime==4.9.* in /home/metju/miniconda3/lib/python3.13/site-packages (from hydra-core->-r /dev/fd/63 (line 2)) (4.9.3)
Requirement already satisfied: colorlog in /home/metju/miniconda3/lib/python3.13/site-packages (from hydra_colorlog->-r /dev/fd/63 (line 3)) (6.10.1)
Requirement already satisfied: python-dateutil>=2.8.2 in /home/metju/miniconda3/lib/python3.13/site-packages (from pandas->-r /dev/fd/63 (line 7)) (2.9.0.post0)
Requirement already satisfied: contourpy>=1.0.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from matplotlib->-r /dev/fd/63 (line 9)) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /home/metju/miniconda3/lib/python3.13/site-packages (from matplotlib->-r /dev/fd/63 (line 9)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from matplotlib->-r /dev/fd/63 (line 9)) (4.61.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from matplotlib->-r /dev/fd/63 (line 9)) (1.4.9)
Requirement already satisfied: pillow>=8 in /home/metju/miniconda3/lib/python3.13/site-packages (from matplotlib->-r /dev/fd/63 (line 9)) (12.1.0)
Requirement already satisfied: pyparsing>=3 in /home/metju/miniconda3/lib/python3.13/site-packages (from matplotlib->-r /dev/fd/63 (line 9)) (3.3.2)
Requirement already satisfied: pyarrow>=21.0.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from datasets->-r /dev/fd/63 (line 10)) (23.0.0)
Requirement already satisfied: dill<0.4.1,>=0.3.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from datasets->-r /dev/fd/63 (line 10)) (0.4.0)
Requirement already satisfied: requests>=2.32.2 in /home/metju/miniconda3/lib/python3.13/site-packages (from datasets->-r /dev/fd/63 (line 10)) (2.32.5)
Requirement already satisfied: xxhash in /home/metju/miniconda3/lib/python3.13/site-packages (from datasets->-r /dev/fd/63 (line 10)) (3.6.0)
Requirement already satisfied: multiprocess<0.70.19 in /home/metju/miniconda3/lib/python3.13/site-packages (from datasets->-r /dev/fd/63 (line 10)) (0.70.18)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /home/metju/miniconda3/lib/python3.13/site-packages (from fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (3.13.3)
Requirement already satisfied: psutil in /home/metju/miniconda3/lib/python3.13/site-packages (from accelerate->-r /dev/fd/63 (line 11)) (7.2.1)
Requirement already satisfied: torch>=2.0.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from accelerate->-r /dev/fd/63 (line 11)) (2.11.0.dev20260128+cu128)
Requirement already satisfied: absl-py>=0.4 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (2.4.0)
Requirement already satisfied: grpcio>=1.48.2 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (1.76.0)
Requirement already satisfied: markdown>=2.6.8 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (3.10.1)
Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (6.33.4)
Requirement already satisfied: setuptools>=41.0.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (82.0.1)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from tensorboard->-r /dev/fd/63 (line 12)) (3.1.5)
Requirement already satisfied: joblib>=1.3.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from scikit-learn->-r /dev/fd/63 (line 14)) (1.5.3)
Requirement already satisfied: threadpoolctl>=3.2.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from scikit-learn->-r /dev/fd/63 (line 14)) (3.6.0)
Requirement already satisfied: click in /home/metju/miniconda3/lib/python3.13/site-packages (from nltk->-r /dev/fd/63 (line 15)) (8.2.1)
Collecting iniconfig>=1.0.1 (from pytest->-r /dev/fd/63 (line 17))
  Using cached iniconfig-2.3.0-py3-none-any.whl.metadata (2.5 kB)
Requirement already satisfied: pluggy<2,>=1.5 in /home/metju/miniconda3/lib/python3.13/site-packages (from pytest->-r /dev/fd/63 (line 17)) (1.5.0)
Requirement already satisfied: pygments>=2.7.2 in /home/metju/miniconda3/lib/python3.13/site-packages (from pytest->-r /dev/fd/63 (line 17)) (2.19.2)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (25.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.10.0,>=2023.1.0->datasets->-r /dev/fd/63 (line 10)) (1.22.0)
Requirement already satisfied: six>=1.5 in /home/metju/miniconda3/lib/python3.13/site-packages (from python-dateutil>=2.8.2->pandas->-r /dev/fd/63 (line 7)) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /home/metju/miniconda3/lib/python3.13/site-packages (from requests>=2.32.2->datasets->-r /dev/fd/63 (line 10)) (3.4.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from requests>=2.32.2->datasets->-r /dev/fd/63 (line 10)) (2.6.1)
Requirement already satisfied: sympy>=1.13.3 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (3.6.1)
Requirement already satisfied: jinja2 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (3.1.6)
Requirement already satisfied: cuda-bindings==12.9.4 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.9.4)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.17.1.4 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (9.17.1.4)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.28.9 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (2.28.9)
Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (3.4.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (1.13.1.3)
Requirement already satisfied: triton==3.6.0+git9844da95 in /home/metju/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (3.6.0+git9844da95)
Requirement already satisfied: cuda-pathfinder~=1.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from cuda-bindings==12.9.4->torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (1.3.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/metju/miniconda3/lib/python3.13/site-packages (from sympy>=1.13.3->torch>=2.0.0->accelerate->-r /dev/fd/63 (line 11)) (1.3.0)
Requirement already satisfied: markupsafe>=2.1.1 in /home/metju/miniconda3/lib/python3.13/site-packages (from werkzeug>=1.0.1->tensorboard->-r /dev/fd/63 (line 12)) (3.0.3)
Downloading ruff-0.15.20-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.5/11.5 MB 67.0 MB/s  0:00:00
Downloading pytest-9.1.1-py3-none-any.whl (386 kB)
Using cached iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Installing collected packages: ruff, iniconfig, pytest

Successfully installed iniconfig-2.3.0 pytest-9.1.1 ruff-0.15.20

## Covariance matrices
Pre-computed large matrix copied from :
- Source: 
- Target: 

Small 5000-sample matrix computed on the cluster:

- Output: 

## Prefix cache
The Qwen3-8B config uses  with .
Generated the cache once so ROME edits can reuse it:

- Output: 

## ROME structural run (N=3, large 100k covariance)

- Run ID: 
- Output root: 
- Model used the precomputed 100k matrix automatically (configured in ).

## Artifact verification
- Execution artifact: 
  - 3/3 cases 
  -  matches target layer 10
  - ROME success metrics: efficacy_score=1.0, paraphrase_score=1.0
- Capture artifact: 
  - 
- Detection analysis:
  - Version: ImageMagick 7.1.1-47 Q16-HDRI x86_64 22763 https://imagemagick.org
Copyright: (C) 1999 ImageMagick Studio LLC
License: https://imagemagick.org/script/license.php
Features: Cipher DPC HDRI Modules OpenMP(4.5) 
Delegates (built-in): bzlib cairo djvu fftw fontconfig freetype gslib gvc heic jbig jng jp2 jpeg jxl lcms lqr ltdl lzma openexr pangocairo png ps raqm raw rsvg tiff webp wmf x xml zip zlib zstd
Compiler: gcc (15.1)
Usage: composite [options ...] image [options ...] composite
  [ [options ...] mask ] [options ...] composite

Image Settings:
  -affine matrix       affine transform matrix
  -alpha option        on, activate, off, deactivate, set, opaque, copy
                       transparent, extract, background, or shape
  -authenticate password
                       decipher image with this password
  -blue-primary point  chromaticity blue primary point
  -colorspace type     alternate image colorspace
  -comment string      annotate image with comment
  -compose operator    composite operator
  -compress type       type of pixel compression when writing the image
  -define format:option
                       define one or more image format options
  -depth value         image depth
  -density geometry    horizontal and vertical density of the image
  -display server      get image or font from this X server
  -dispose method      layer disposal method
  -dither method       apply error diffusion to image
  -encoding type       text encoding type
  -endian type         endianness (MSB or LSB) of the image
  -filter type         use this filter when resizing an image
  -font name           render text with this font
  -format "string"     output formatted image characteristics
  -gravity type        which direction to gravitate towards
  -green-primary point chromaticity green primary point
  -interlace type      type of image interlacing scheme
  -interpolate method  pixel color interpolation method
  -label string        assign a label to an image
  -limit type value    pixel cache resource limit
  -matte               store matte channel if the image has one
  -monitor             monitor progress
  -page geometry       size and location of an image canvas (setting)
  -pointsize value     font point size
  -quality value       JPEG/MIFF/PNG compression level
  -quiet               suppress all warning messages
  -red-primary point   chromaticity red primary point
  -regard-warnings     pay attention to warning messages
  -respect-parentheses settings remain in effect until parenthesis boundary
  -sampling-factor geometry
                       horizontal and vertical sampling factor
  -scene value         image scene number
  -seed value          seed a new sequence of pseudo-random numbers
  -size geometry       width and height of image
  -support factor      resize support: > 1.0 is blurry, < 1.0 is sharp
  -synchronize         synchronize image to storage device
  -taint               declare the image as modified
  -transparent-color color
                       transparent color
  -treedepth value     color tree depth
  -tile                repeat composite operation across and down image
  -units type          the units of image resolution
  -verbose             print detailed information about the image
  -virtual-pixel method
                       virtual pixel access method
  -white-point point   chromaticity white point

Image Operators:
  -blend geometry      blend images
  -border geometry     surround image with a border of color
  -bordercolor color   border color
  -channel mask        set the image channel mask
  -colors value        preferred number of colors in the image
  -decipher filename    convert cipher pixels to plain pixels
  -displace geometry   shift lookup according to a relative displacement map
  -dissolve value      dissolve the two images a given percent
  -distort geometry    shift lookup according to a absolute distortion map
  -encipher filename   convert plain pixels to cipher pixels
  -extract geometry    extract area from image
  -geometry geometry   location of the composite image
  -identify            identify the format and characteristics of the image
  -monochrome          transform image to black and white
  -negate              replace every pixel with its complementary color 
  -profile filename    add ICM or IPTC information profile to image
  -quantize colorspace reduce colors in this colorspace
  -repage geometry     size and location of an image canvas (operator)
  -rotate degrees      apply Paeth rotation to the image
  -resize geometry     resize the image
  -sharpen geometry    sharpen the image
  -shave geometry      shave pixels from the image edges
  -stegano offset      hide watermark within an image
  -stereo geometry     combine two image to create a stereo anaglyph
  -strip               strip image of all profiles and comments
  -thumbnail geometry  create a thumbnail of the image
  -transform           affine transform image
  -type type           image type
  -unsharp geometry    sharpen the image
  -watermark geometry  percent brightness and saturation of a watermark
  -write filename      write images to this file

Image Stack Operators:
  -swap indexes        swap two images in the image sequence

Miscellaneous Options:
  -debug events        display copious debugging information
  -help                print program options
  -list type           print a list of supported option arguments
  -log format          format of debugging information
  -version             print version information

By default, the image format of 'file' is determined by its magic
number.  To specify a particular image format, precede the filename
with an image format name and a colon (i.e. ps:image) or specify the
image type as the filename suffix (i.e. image.ps).  Specify 'file' as
'-' for standard input or output.: accuracy 1.0 (detected layer 10 on all 3 cases)
  - : accuracy 0.0 (detected different layers; detector behavior, not artifact error)

## Rendered graphs
Renderer preset  produced all 7 graph outputs under :
1.  – machine-readable paper summary ()
2.  – detection accuracy summary + plot
3.  – per-layer/window CSV/JSON/PNG
4.  – per-case detector signal PNGs (cases 0,1,2)
5.  – ROME success-rate CSV/JSON/PNG
6.  – run-level aggregate JSON
7.  – 5x4 per-layer artifact grid PNG/PDF/JSON

## Files to bring back to local PC
-  (manifest, artifacts, graphs)
- 
- 
- 
