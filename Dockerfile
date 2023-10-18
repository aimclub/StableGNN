FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install required packages
RUN set -xe \
 && apt-get -y update \
 && apt-get install -fyqq software-properties-common curl build-essential git libaio-dev llvm-10 clang wget \
 && apt-get -y update \
 && add-apt-repository universe \
 && apt-get -y update \
 && apt-get -fyqq install python3.9-full python3.9-dev python3-pip \
 && apt-get clean

# Let's upgrade pip first
RUN set -xe \
 && python3.9 -m pip install --upgrade pip

# Install python packages
RUN set -xe \
    # PyTorch MUST BE installed first
 && python3.9 -m pip install 'torch==1.12.1+cu116' -f https://download.pytorch.org/whl/torch_stable.html
RUN set -xe \
    # And only then all other dependencies
 && python3.9 -m pip install 'torch-geometric==2.0.4' 'torch-sparse==0.6.15' 'torch-scatter==2.1.1' -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
COPY requirements.txt ./
RUN set -xe \
 && python3.9 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
# Install Jupyter
RUN set -xe \
 && python3.9 -m pip install jupyter
RUN set -xe \
 && jupyter notebook --generate-config \
 && echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_notebook_config.py \
 && echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py \
 && echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_notebook_config.py \
 && echo "c.NotebookApp.password = u'sha1:fd40b23609dd:882af6cdf722657245be6f4abd9b641a84ef9c2a'" >> /root/.jupyter/jupyter_notebook_config.py

# Copy project files
COPY . .

ENTRYPOINT ["/app/entrypoint.sh"]
