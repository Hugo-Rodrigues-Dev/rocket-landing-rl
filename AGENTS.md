# Règles pour les assistants de code IA

Ce dépôt est un projet étudiant à trois. Chaque étudiant (Hugo, Quentin, Mathis) construit son propre environnement et ses propres agents RL dans son dossier. Ces règles s'appliquent à tout assistant (Claude Code, Codex, Cursor…). Lis aussi [`CONTRACT.md`](CONTRACT.md) avant d'écrire du code.

## Pour qui tu travailles

L'étudiant est donné par le préfixe de la branche courante (`hugo/…`, `quentin/…`, `mathis/…`). En cas de doute, demande. Lis ensuite `players/<prenom>/AGENTS.md` s'il existe : ce sont ses consignes personnelles, et elles complètent celles-ci.

## Règles

1. **Travaille uniquement dans `players/<prenom>/`.** Ne modifie jamais le dossier d'un autre étudiant. Ne touche aux fichiers communs (`CONTRACT.md`, `AGENTS.md`, `README.md`, `docs/`, `pyproject.toml` racine, `uv.lock` sauf par `uv add`, `.github/`) que si l'étudiant le demande explicitement.
2. **Respecte la pile du sujet** : Python, Gymnasium, PyTorch, Stable-Baselines3, WandB, MuJoCo. Demande avant d'ajouter une dépendance, puis ajoute-la avec `uv add --package rocket_<prenom> <paquet>`.
3. **Respecte le contrat.** Ne change jamais le scénario fixé, les seuils ou les graines d'évaluation (10 000 à 10 099) pour améliorer un résultat. N'entraîne jamais sur ces graines.
4. **Rapporte les résultats honnêtement** : plusieurs graines d'entraînement, pas de sélection du meilleur run présenté comme typique, les échecs compris.
5. **Git** : ne pousse jamais sur `main`, ne force jamais un push, ne modifie pas l'identité git. Pas de ligne `Co-Authored-By` ni de mention d'IA dans les commits et les PR.
6. **Ne committe rien de lourd ni de secret** : `.venv/`, `wandb/`, `runs/`, vidéos, checkpoints intermédiaires, clés API. Seul le modèle final d'une phase va dans `players/<prenom>/models/` (moins de 20 Mo).
7. **Vérifie avant de rendre la main** : `uv run ruff check .` et `uv run pytest players/<prenom>` doivent passer.

## Commandes utiles

```bash
uv sync --all-packages            # installer
uv run pytest players/<prenom>    # tests de l'étudiant
uv run ruff check . --fix         # style
```
