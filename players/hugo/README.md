# Hugo

Environnement et agents de Hugo. Le [contrat](../../CONTRACT.md) définit les scénarios et leur évaluation.

Le [guide pas à pas](GUIDE.md) explique le code actuel, les outils, la boucle d'apprentissage et les tests et rejeux prévus à chaque étape.

Pour l'expérience sur les départs complets, voir la [comparaison PPO/SAC par phase](comparaisons/README.md).
Les réglages des deux algorithmes vivent dans `algorithms.py`. Ils partagent le simulateur `env.py`,
le protocole `phase0.py`, l'entraînement `train_phase0.py` et l'évaluation `evaluate_phase0.py`.

## Phase 0 complète, pilotage assisté

PPO et SAC franchissent le seuil du contrat avec **98 poses sur 100 chacun**. Les réseaux
apprennent des corrections sur les trois commandes d'un pilote manuel. Le pilote seul obtient
97 poses et la meilleure précision. Les modèles évalués sont dans `models/`. Voir la
[fiche de phase 0](comparaisons/phase-0.md) pour les métriques, les résultats par graine et les
commandes pour évaluer ou rejouer PPO et SAC. Le mode réseau seul reste à mesurer.

Pour comparer les **trois pilotes dans une seule fenêtre**, depuis la racine avec la `.venv` locale :

```powershell
$env:PYTHONPATH = (Resolve-Path players/hugo/src).Path
.\.venv\Scripts\python.exe -m rocket_hugo.demo_3d --mode compare --seed 10046
```

La comparaison démarre en pause. Les touches affichées en bas contrôlent le rejeu :

| Touche | Effet |
|---|---|
| `1` / `2` / `3` | Pilote seul / pilote + PPO / pilote + SAC. Les trois repartent du même état. |
| `D` | Alterne entre les départs 10 046 et 7 200. Réinitialise les trois résultats affichés. |
| `V` / `B` | Recule / avance parmi les versions d'entraînement enregistrées et recharge le vol en pause. |
| `P` / `N` / `R` | Lecture ou pause / un pas de simulation / rejouer le vol courant. |

La **graine de départ** fixe la position, la vitesse et l'orientation de la fusée au début du
vol. `--seed` permet de choisir d'autres départs. Le **nombre de décisions apprises** sélectionne
une version de PPO ou SAC. Il ne modifie pas le pilote seul ni l'état initial du vol.
Le visualiseur joue les commandes de façon déterministe : rejouer le même pilote, le même départ
et la même version donne le même vol. Le tableau à droite compare les résultats au point
d'entraînement sélectionné.

Les versions intermédiaires sont des sauvegardes à **0, 2 048, 4 096 et 6 144 décisions**.
La version finale des deux modèles de phase 0 correspond à **8 192 décisions**. Même à zéro,
PPO et SAC ajoutent une correction issue d'un réseau initialisé à la commande du pilote manuel,
qui sait déjà atterrir dans beaucoup de cas. Un premier contact réussi ne prouve donc pas
que le réseau a appris quoi que ce soit.

Pour choisir une version et un départ depuis le terminal :

```powershell
.\.venv\Scripts\python.exe -m rocket_hugo.demo_3d --mode compare --seed 7200 --checkpoint 2048
```

Sur le départ 10 046, à 8 192 décisions, seul le pilote de référence échoue. Le départ 7 200
permet de comparer trois poses. Ajoute `--headless` pour jouer les trois vols dans le terminal.

Les sauvegardes intermédiaires restent dans `runs/`. Sur une autre copie du dépôt, on peut
recréer l'historique dans les dossiers attendus par le visualiseur :

```powershell
.\.venv\Scripts\python.exe -m rocket_hugo.train_phase0 --algo ppo --seed 2 --snapshot-every 2048 --out players/hugo/runs/phase0-history-ppo
.\.venv\Scripts\python.exe -m rocket_hugo.train_phase0 --algo sac --seed 1 --snapshot-every 2048 --out players/hugo/runs/phase0-history-sac
```

Le visualiseur indique les versions effectivement disponibles. Sans les dossiers d'historique,
il reste possible de comparer les deux modèles finaux dans `models/`.

Le mode `compare` est la vue **phase 0**, avec barge fixe et modèles propres à cette phase.
Quand on ajoutera la phase 1, on pourra réutiliser la même fenêtre avec un choix de phase :
la scène, les modèles et les mesures viendront alors de la phase choisie. La vue phase 0
restera accessible avec ses propres départs et versions.

## Rejouer le premier apprentissage PPO

Depuis la racine du dépôt, avec PyTorch et Stable-Baselines3 installés dans `.venv` :

```powershell
$env:PYTHONPATH = (Resolve-Path players/hugo/src).Path
.\.venv\Scripts\python.exe -m rocket_hugo.replay_vertical --run players/hugo/runs/vertical-ppo-s0 --seed 5100
```

La fenêtre s'ouvre en pause. Appuie sur `P` pour jouer un vol entier, ou sur `N` pour observer
chaque décision. `1` sélectionne le réseau avant entraînement, `2` à `5` les versions successives.
`R` rejoue la version courante. Tous les réseaux voient le même départ. Pour obtenir les issues
dans le terminal, ajoute `--headless`.

Sur cet exercice, l'agent part entre 1 et 6 m du pont et n'observe que la hauteur et la vitesse.
Il ne contrôle que la poussée. Une arrivée à moins de 0,2 m/s compte comme un contact doux.
La scène 3D affiche la trajectoire calculée par le modèle vertical analytique qui a servi à
l'entraînement. L'évaluation de la phase 0 complète utilisera le simulateur 3D et ses 100 graines.

Les versions et les résultats sont dans `players/hugo/runs/`, un dossier local ignoré par Git.
Pour reproduire un essai dans un nouveau dossier :

```powershell
.\.venv\Scripts\python.exe -m rocket_hugo.train_vertical --seed 0 --out players/hugo/runs/vertical-ppo-repro
```

Premier essai PPO sur CPU, trois graines d'entraînement, réseau de deux couches cachées de
32 unités et collecte par blocs de 512 décisions. Les nombres comptent les contacts **strictement
en dessous de 0,2 m/s** sur les mêmes 20 départs de développement.

| Décisions par modèle | Graine 0 | Graine 1 | Graine 2 |
|---:|---:|---:|---:|
| 0 | 0/20 | 0/20 | 0/20 |
| 8 192 | 0/20 | 0/20 | 0/20 |
| 16 384 | 0/20 | 0/20 | 0/20 |
| 24 576 | 20/20 | 20/20 | 20/20 |
| 32 768 | 20/20 | 0/20 | 20/20 |

Sur 20 autres départs (graines 6 100 à 6 119), les trois modèles finaux obtiennent respectivement
20/20, 0/20 et 20/20. Pour la graine 1, la vitesse moyenne finale vaut environ 0,2002 m/s, juste
au-dessus du seuil. Pour voir cette régression dans le visualiseur, ouvre `vertical-ppo-s1`
et compare les touches `4` et `5` sur le même départ. Les tests du contrat restent à part.

## Voir la scène 3D de Hugo

Depuis la racine, après `uv sync --all-packages --locked` :

```powershell
uv run --locked python -m rocket_hugo.demo_3d --mode brake
uv run --locked python -m rocket_hugo.demo_3d --mode freefall
uv run --locked python -m rocket_hugo.demo_3d --mode gimbal
```

La fenêtre suit la descente et affiche la hauteur, la vitesse verticale, l'inclinaison,
la poussée et la tuyère. Les panneaux techniques de MuJoCo sont masqués à l'ouverture.
Les raccourcis ci-dessous s'utilisent dans la fenêtre 3D, y compris après la fin du vol.

| Touche | Expérience | Option équivalente |
|---|---|---|
| `1` | Chute libre, moteur coupé. | `--mode freefall` |
| `2` | La poussée compense le poids. La fusée conserve sa vitesse de descente. | `--mode weight` |
| `3` | Freinage vertical avec une règle manuelle. | `--mode brake` |
| `4` | Départ incliné de 5°, freinage et tuyère centrée. | `--mode tilt-open` |
| `5` | Même départ, avec une correction de l'angle sur un axe. | `--mode tilt-stabilize` |
| `R` | Rejouer le scénario courant depuis son départ. | |
| `P` | Mettre en pause ou reprendre. | |
| `N` | Passer en pause et avancer d'une décision, soit 0,05 s de physique en vol. | |

`--mode gimbal` applique une tuyère fixe pour illustrer la rotation. `--mode random --seed 12`
joue des commandes aléatoires sur un départ du contrat. Ajoute `--headless` pour calculer
le résultat sans fenêtre. Ferme la fenêtre pour revenir au terminal.

Pour lancer **seulement la simulation et son visualiseur**, le moteur d'entraînement et CUDA ne sont pas
nécessaires. Cette commande utilise les versions de MuJoCo et Gymnasium du verrou sans installer tout le dépôt :

```powershell
Push-Location players/hugo/src
try {
    uv run --no-project --python 3.12 --with mujoco==4.0.1 --with gymnasium==1.4.0 python -m rocket_hugo.demo_3d --mode brake
} finally {
    Pop-Location
}
```

Pour inspecter la scène seule, ouvre [MuJoCo Studio dans le navigateur](https://live.mujoco.org/)
et dépose `players/hugo/src/rocket_hugo/assets/scene.xml`. La scène s'ouvre avec la gravité et les contacts.
Le programme Python `demo_3d` est nécessaire pour voir les scénarios de freinage et de tuyère.

## Première expérience : descente verticale

Depuis la racine du dépôt, après `uv sync --all-packages --locked` :

```powershell
uv run --locked python -m rocket_hugo.demo_vertical
uv run --locked pytest players/hugo
```

La démo compare trois commandes depuis le même départ : 100 m au-dessus du pont, à −10 m/s.
Le modèle vertical utilise uniquement la bibliothèque standard et peut aussi se lancer depuis la racine sans
installer la pile d'entraînement :

```powershell
Push-Location players/hugo/src
try { python -m rocket_hugo.demo_vertical } finally { Pop-Location }
```

| Commande | Effet sur la vitesse |
|---|---|
| `−1` | Moteur coupé. La gravité accélère la descente. |
| `0` | Poussée de 9 810 N. La vitesse reste à −10 m/s car la poussée compense exactement le poids. |
| Freinage | Une règle ajuste la poussée pour suivre une consigne de descente qui ralentit près du pont. |

À chaque décision, la règle lit la hauteur et la vitesse, puis choisit une commande entre −1 et +1.
Le modèle calcule l'accélération `poussée / masse − g`, puis la nouvelle hauteur et la nouvelle vitesse.
La commande dure 0,05 s. Si le pont est atteint entre deux décisions, le modèle s'arrête au premier contact
et affiche la vitesse juste avant l'impact.

Ce modèle sert de référence pour vérifier les unités et les trajectoires. L'altitude et la vitesse verticale
suffisent pour cette expérience. La phase 0 du contrat nécessite aussi les mouvements latéraux, l'attitude,
les quatre contacts et les règles complètes de fin d'épisode. Elle reste à implémenter et à valider.

## Où lire le code

| Emplacement | Rôle |
|---|---|
| `src/rocket_hugo/vertical.py` | État vertical, conversion de la commande en newtons, transition et premier contact. |
| `src/rocket_hugo/demo_vertical.py` | Départ fixe, trois scénarios et affichage. Le pilote de démo vit ici. |
| `src/rocket_hugo/assets/scene.xml` | Modèle MuJoCo de Hugo : fusée libre, mer, barge fixe, quatre jambes et points de mesure. |
| `src/rocket_hugo/env.py` | Départs, poussée et tuyère, observations, contacts, issues et récompenses de phase 0. |
| `src/rocket_hugo/demo_3d.py` | Rejoue des commandes prédéfinies dans le même environnement Gymnasium que le futur agent. |
| `tests/test_vertical.py` | Vérifications du mouvement et du calcul du contact. |
| `tests/test_env.py` | Vérifie Gymnasium, les commandes, la reproductibilité, le contact et la troncature. |
| `tests/test_setup.py` | Vérifie les imports du projet et la chute libre dans MuJoCo. |
| `pyproject.toml` | Déclare le paquet `rocket_hugo` et ses dépendances dans l'espace de travail `uv`. |
| `models/` | Accueillera les modèles finaux des phases validées. |

`reset()` tire un départ du contrat par défaut. Le départ vertical centré de la démo est une option de
diagnostic. `step(action)` renvoie l'observation suivante, la récompense et l'issue de l'épisode.
La performance d'un agent sur les 100 départs du contrat reste à mesurer.
