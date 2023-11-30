This is the code for a class project for the course EEPS 587: `Geospatial Science` at the Washington University in St. Louis. In this, we evaluate the representation space of `GeoCLAP` for a downstream task of land-type classification. `GeoCLAP` is a self-supervised framework which learns a common represntation space between three modalities: `satellite imagery`, `audio`, and `text`.\
[Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping, BMVC 2023](https://arxiv.org/abs/2309.10667)

1. Clone this repo
    ```
    git clone git@github.com:mvrl/geoclap.git
    cd geoclap
    ```
2. Environment
    ```
    We use the docker image `ksubash/geoclap:latest` to execute our code.

    ```


4. Check `config.py`  to setup relevant paths for data and pretrained checkpoints. 

