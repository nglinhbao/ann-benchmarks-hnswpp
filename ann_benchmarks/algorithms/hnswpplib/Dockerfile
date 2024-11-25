FROM ann-benchmarks

RUN apt-get install -y python-setuptools python-pip
RUN pip3 install pybind11 numpy setuptools
RUN git clone https://github.com/nglinhbao/hnswpplib.git;cd hnswpplib; git checkout main

RUN cd hnswpplib/python_bindings; python3 setup.py install

RUN python3 -c 'import hnswpplib'

