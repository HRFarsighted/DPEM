# DPEM:Dual-Perspective Enhanced Mamba for Multivariate Time Series Forecasting
## DPEM Model Architecture
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

```
scikit-learn==1.3.0

numpy==1.26.4

matplotlib==3.7.0

torch==2.0.1

reformer-pytorch==1.4.4

mamba-ssm==1.2.0
```
### Datasets
Detailed dataset descriptions. $Dim$ denotes the variate number of each dataset. $Dataset$ $Size$ denotes the total number of time points in (Train, Validation, Test) split respectively. $Prediction$ $Length$ denotes the future time points to be predicted. $Frequency$ denotes the sampling interval of time points.
![image](https://github.com/user-attachments/assets/d4134d1b-d261-43f2-8fba-782bc4de4a4e)

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

