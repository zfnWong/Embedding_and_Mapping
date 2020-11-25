## Embedding & Mapping by pytorch



## Dataset

Gowalla, Yelp2018 and Amazon-book and one small dataset LastFM.


## Run

* change base directory - `ROOT_PATH` in `code/world.py`

* command

```
python main.py --model lgn_ecc --n_layers 2 --dataset gowalla --test_batch 500 --fine_tune_epochs 500
```
```
python main.py --model ngcf --n_layers 3 --dataset lastfm --epochs 550 --simutaneously 1 --test_every_n_epochs 10
```
```
python main.py --model lgn_ecc --n_layers 2 --dataset lastfm --epochs 600  --test_every_n_epochs 10 --ecc 0 --ecc_layer 1 --fine_tune_epochs 300 --p_dist 1
```


