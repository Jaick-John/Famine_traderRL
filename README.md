# Famine Trader (RL)

Reinforcement-learning trading agent for food commodities under supply/famine shocks.

## Setup

```
git clone https://github.com/Jaick-John/Famine_traderRL.git
cd Famine_traderRL
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
```

## Real data

Put these three files in `data/raw/` (the folder is gitignored):

| File | Source |
|---|---|
| `wheat_zw.csv` | Wheat futures `ZW=F` from yfinance, columns `Date,Close` |
| `fao_ffpi.csv` | FAO Food Price Index CSV (monthly, includes `Cereals`) |
| `oni.txt` | NOAA CPC ONI: `https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt` |

`configs/ppo_default.yaml` points `data_source` at `data/raw`. Data is split by
date: train 60% / val 15% / test 25%. Training uses train, evaluation during
training uses val, and `analysis/compare.py` and `demo/demo_run.py` use test.
Set `data_source: null` to use synthetic data instead.

## Structure

- `envs/` gymnasium environment (owner: Jaick)
- `agents/` baseline and RL agents
- `data/` download/cleaning scripts (raw data is not committed)
- `configs/` YAML configs
- `experiments/` training outputs (not committed)
- `analysis/` plots and result analysis
- `tests/` pytest tests
- `demo/` demo material

## Contributing

1. Never commit to `main`. Create a branch: `git switch -c feature/<your-name>-<task>`
2. Only edit the files for your assigned task.
3. Run `pytest` before pushing.
4. Push your branch and open a Pull Request. Jaick reviews and merges.
5. Don't commit data files, `.venv`, or model outputs.
