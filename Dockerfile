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

# Install python packages
RUN set -xe \
 && python3.9 -m pip install --upgrade pip \
 && python3.9 -m pip install 'torch==1.12.1+cu116' 'torchvision==0.13.1' 'torchaudio==0.12.1' 'torch-geometric==2.3.1' 'torch-sparse==0.6.17' 'torch-scatter==2.1.1' -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
COPY requirements.txt ./
RUN set -xe \
 && python3.9 -m pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

CMD ["sleep", "inf"]
