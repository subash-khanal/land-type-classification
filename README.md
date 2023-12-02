This is the code for a project: `Evaluating GeoCLAP embeddings for land-type classification`. This work was carried out for the course `EEPS 587: Geospatial Science` at the Washington University in St. Louis during Fall 2023. 

`GeoCLAP` is a self-supervised framework which learns a common represntation space between three modalities: `satellite imagery`, `audio`, and `text`.\
Paper: [Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping, BMVC 2023](https://arxiv.org/abs/2309.10667)


Note:  We use the docker image `ksubash/geoclap:latest` to run all experiments in this project.
1. Clone this repo
    ```
    git clone git@github.com:subash-khanal/land-type-classification.git
    ```

2. Follow instructions in the publicly available code for [GeoCLAP](https://github.com/mvrl/geoclap.git) to download [GeoCLAP pre-trained checkpoints](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link). Once downloaded, their paths should be set up in the following `config ` variables:\
   `cfg.sentinel_ckpt_path`,\
   `cfg.googleEarth_ckpt_path` and \
   `cfg.satmae_pretrained_ckpt`

3. We use three land-type datasets in our experiments: [NWPU_RESISC45](https://figshare.com/articles/dataset/NWPU-RESISC45_Dataset_with_12_classes/16674166
), [RSSCN7](https://github.com/palewithout/RSSCN7), and [UCMerced_LandUse](http://weegee.vision.ucmerced.edu/datasets/landuse.html). Please download the data following the provided hyperlinks and set paths accordingly in the `config.py`.

4. Create `train/val/test` split for these three datasets using:
    ```
    python -m land-type-classification.data_prep.data_split
    ```

5. Now, linear probe related experiments can be run in using:
    ```
    python -m land-type-classification.train --wandb_mode online --mode train --run_name $dataset_$sat_type --dataset_type $dataset  --sat_type $sat_type
    ```

    For command to run all 6 experiments (2 geoclap checkpoints X 3 datasets) please refer to the file `experiments.txt`.

6. The best saved checkpoints from step $5$, are then evaluated on each dataset's test set using:
    ```
    python -m land-type-classification.linear_probe_eval --dataset_type $dataset  --sat_type $sat_type --lc_ckpt_path "path-to-the-best-lc checkpoint"
    ```

7. We can also evaluate the original GeoCLAP model in zero-shot fashion by querying with dataset specific label query set. This can be done using:
    ```
    python -m land-type-classification.zero_shot_eval --dataset_type $dataset  --sat_type $sat_type
    ```
    Note: To see how we create prompts for the classes of our dataset, refer to file `utilities/landtype_text_prompts.py`.

8. Finally, for a `.csv` file containing ground truth and predicted results for the test set obtained from either step $6$ or $7$, we can obtain confusion matrix using:

    ```
    python -m land-type-classification.get_confusion_matrix --results_path "path-to-results-csv"
    ```

    Note: The packages required for generating confusion matrix were not available in the `geoclap` docker environment, so step $8$ has to be executed outside this docker environment where following packages: `matplotlib`,`seaborn`,`sklearn` are installed.