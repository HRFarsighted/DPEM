# DPEM:Dual-Perspective Enhanced Mamba for Multivariate Time Series Forecasting
![模型图](https://github.com/user-attachments/assets/4ea7d8f0-306e-4cbd-8241-de278e360a08)


## Contributions:

-- A Dual-Perspective Enhanced Mamba DPEM framework is introduced for effective multivariate time series forecasting.
-- DPEM utilizes a dual-perspective approach to separately model temporal dependencies and spatial dependencies, enabling comprehensive spatiotemporal feature extraction.
-- We designed a cross attention mechanism that enables bidirectional interactions between temporal-spatial and spatial-temporal features, resulting in a deeper fusion of the spatiotemporal dependencies.
-- Extensive experiments conducted on 8 real-world datasets demonstrate superior performance compared to SOTA baseline models.

## Getting Start:
### Installation
```bash
pip install -r requirements.txt
```
### Datasets


### Train and evaluate

```bash
# Example: Exchange
bash ./scripts/multivariate_forecasting/Exchange/exchange_96.sh

```


## Acknowledgement:

We are grateful for the following awesome projects when implementing DPEM:

- [iTransformer](https://github.com/thuml/iTransformer)
- [Mamba](https://github.com/state-spaces/mamba)
- [S-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba)
- [Time-Series-Library] (https://github.com/thuml/Time-Series-Library)

## Citation  

