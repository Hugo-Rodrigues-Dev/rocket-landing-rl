# Rocket Landing RL

Projet IA & RL (CPE Lyon, 5A). Une fusée doit apprendre à se poser sur une barge qui dérive en mer, par apprentissage par renforcement.

Nous sommes trois. Chacun construit **son propre environnement et ses propres agents**, à son rythme. Un contrat commun fixe le problème et les critères de réussite, pour que les trois approches restent comparables.

| Document | Contenu |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | Le problème, les phases, les critères de réussite, les mesures. **À lire en premier.** |
| [`AGENTS.md`](AGENTS.md) | Les règles pour les assistants de code IA (Claude Code, Codex, Cursor…). |
| [`docs/index.html`](docs/index.html) | Le sujet officiel du projet. |
| [`docs/mdp-etape-1.pdf`](docs/mdp-etape-1.pdf) | Le MDP de référence de la phase 1. |

## Organisation

```
CONTRACT.md, AGENTS.md, docs/   commun : modifié seulement par PR validée à trois
pyproject.toml, uv.lock          commun : l'espace de travail Python
players/
  hugo/      → Hugo
  quentin/   → Quentin
  mathis/    → Mathis
```

- Chacun travaille **dans son dossier** `players/<prenom>/` et **sur ses branches** `<prenom>/<sujet>` (par exemple `hugo/env-phase0`).
- On intègre sur `main` par **pull request**. Une PR personnelle ne touche que son propre dossier.
- Toute modification d'un fichier commun (contrat, dépendances, CI) passe par une PR relue par les deux autres.

## Démarrer

Il faut [uv](https://docs.astral.sh/uv/). Il installe Python 3.12 tout seul si besoin.

```bash
git clone git@github.com:Hugo-Rodrigues-Dev/rocket-landing-rl.git
cd rocket-landing-rl
uv sync --all-packages      # crée .venv avec les trois paquets et les dépendances
uv run pytest               # vérifie que MuJoCo, Gymnasium, SB3, PyTorch et WandB s'importent
uv run ruff check .         # style du code
wandb login                 # une fois, avec sa clé personnelle
```

Chaque dossier joueur est un paquet Python : `rocket_hugo`, `rocket_quentin`, `rocket_mathis`. Le code va dans `players/<prenom>/src/rocket_<prenom>/`, les tests dans `players/<prenom>/tests/`.

Pour ajouter une dépendance à son seul paquet :

```bash
uv add --package rocket_hugo moviepy
```

## Pile imposée par le sujet

Python, **Gymnasium**, **PyTorch**, **Stable-Baselines3**, **WandB**. Nous ajoutons **MuJoCo** pour la physique 3D.

## Commits

Le sujet demande **nos noms complets et nos adresses CPE** dans les commits. À régler une fois dans le dépôt :

```bash
git config user.name "Prénom Nom"
git config user.email "prenom.nom@cpe.fr"
```
