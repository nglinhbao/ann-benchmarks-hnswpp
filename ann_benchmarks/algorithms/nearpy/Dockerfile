FROM ann-benchmarks

RUN apt-get update && apt-get install -y libhdf5-openmpi-dev
RUN pip3 install nearpy bitarray redis scikit-learn
RUN python3 -c 'import nearpy'