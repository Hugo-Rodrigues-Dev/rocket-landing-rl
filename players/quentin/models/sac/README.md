# Modèles SAC

Les entraînements et les checkpoints intermédiaires sont dans
`players/quentin/runs/sac/`. Chaque checkpoint Stable-Baselines3 se
charge avec son fichier de normalisation. Le fichier `model.zip` doit
rester avec `vec_normalize.pkl` dans le même dossier.

Pour évaluer un run sur les départs de développement complets, depuis la
racine du dépôt, remplace `$run` par le chemin affiché au lancement.

```powershell
uv run --locked python -m rocket_quentin.evaluate --phase 4 --development-level 1 --run-dir "$run"
```
