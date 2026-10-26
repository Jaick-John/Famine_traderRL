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
