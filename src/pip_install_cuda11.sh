python -m pip install -e `pwd`/`dirname "$0"`
python -m pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html -r `pwd`/`dirname "$0"`/requirements_cuda11.txt
