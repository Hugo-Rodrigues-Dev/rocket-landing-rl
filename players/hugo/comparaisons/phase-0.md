# Phase 0 — PPO et SAC sur barge fixe

## Protocole

Les deux algorithmes volent dans le même simulateur 3D sur les départs complets. Dans cette
expérience, ils utilisent le même **pilote de référence** et apprennent une **correction** sur les
trois commandes. Les corrections maximales sont 0,12 sur la poussée et 0,22 sur chaque commande
normalisée de tuyère, avant saturation. Le pilote seul reste un point de comparaison. Le mode
`direct` existe pour une future expérience où le réseau choisira toute la commande.

La récompense est commune : score de fin du simulateur, petit coût par décision et pour l'écart
latéral, et bonus pour une pose précise et lente. Les mesures du contrat sont calculées sur les
issues physiques et les contacts. Elles restent comparables aux mesures du pilote seul.

| Paramètre | PPO | SAC |
|---|---|---|
| Collecte | Blocs de 512 décisions, puis cinq passes de mise à jour | Mémoire de transitions ; début des mises à jour à 1 024 décisions |
| Fréquence des mises à jour | Une série de mises à jour après chaque bloc | Une mise à jour toutes les quatre décisions une fois la mémoire initialisée |
| Réseau, lot, taux d'apprentissage, actualisation | Deux couches de 32, lot 128, 0,0003, 0,995 | Mêmes valeurs ; SAC emploie ses propres réseaux de valeur et sa régulation de l'exploration |
| Budget et graines | 8 192 décisions par graine, graines 0, 1 et 2 | Identiques |

Les modèles sont comparés sur les graines de développement 7 200 à 7 219. Le choix se fait par
nombre de poses, puis distance médiane au centre, puis ordre des graines en cas d'égalité exacte.
On recharge seulement le candidat choisi de chaque algorithme pour l'évaluation officielle
(100 graines de 10 000 à 10 099, prédiction déterministe). Les graines officielles sont exclues de
l'entraînement et du choix des candidats.

## Développement

Chaque cellule montre le nombre de poses sur 20 et la distance médiane au centre.

| Graine d'entraînement | PPO | SAC |
|---:|---:|---:|
| 0 | 20/20 · 0,275 m | 20/20 · 0,396 m |
| 1 | 20/20 · 0,519 m | **20/20 · 0,356 m** |
| 2 | **20/20 · 0,188 m** | 20/20 · 0,583 m |

Les candidats retenus sont PPO graine 2 et SAC graine 1. Le pilote seul obtient 20/20 et une
distance médiane de 0,168 m sur le même lot. Les 20 poses suffisent à constater que chaque
réseau fonctionne avec le pilote sur ces départs, mais elles rendent la précision nécessaire
pour les départager.

## Évaluation officielle

| Politique | Posées / 100 | Distance médiane sur les poses | Vitesse verticale moyenne au premier contact | Inclinaison moyenne au premier contact | Échecs |
|---|---:|---:|---:|---:|---|
| Pilote seul | 97 | 0,114 m | −1,350 m/s | 0,947° | 2 hors zone, 1 impact dur |
| Pilote + PPO, graine 2 | **98** | 0,273 m | −1,353 m/s | 0,954° | 1 hors zone, 1 impact dur |
| Pilote + SAC, graine 1 | **98** | 0,492 m | −1,269 m/s | 1,014° | 2 hors zone |

Les deux réseaux dépassent le seuil de 95 poses avec le pilote. Ils réussissent les **mêmes
98 départs**. Par rapport au pilote seul, les deux récupèrent le départ 10 046 et réussissent
tous les départs qu'il réussit. PPO est plus précis que SAC sur les poses. SAC touche un peu
plus lentement en moyenne, mais ses deux épisodes hors zone ne font pas partie de cette moyenne.
Le pilote seul reste le plus précis au centre. Ces écarts décrivent ce protocole et ces candidats.
Ils ne démontrent pas qu'un des deux algorithmes est généralement supérieur, ni qu'un réseau
commandant seul la fusée atteint le même résultat.

Les [données par départ](donnees-phase-0.json) permettent de refaire les comparaisons appariées.

## Reproduire

Depuis la racine, la pile peut être installée avec `uv sync --all-packages --locked`. Pour utiliser
la `.venv` locale avec CPU et les sources de travail dans PowerShell :

```powershell
$env:PYTHONPATH = (Resolve-Path players/hugo/src).Path
foreach ($algorithm in 'ppo', 'sac') {
    foreach ($trainingSeed in 0, 1, 2) {
        .\.venv\Scripts\python.exe -m rocket_hugo.train_phase0 --algo $algorithm --seed $trainingSeed --out "players/hugo/runs/repro-phase0-$algorithm-s$trainingSeed"
    }
}
```

Chaque dossier contient `model.zip` et `development.json`, dont les résultats détaillés par
épisode. `--wandb` active un journal local hors ligne, avec `group=hugo` et les tags `phase0` et
algorithme. Les essais peuvent être comparés à plus grande échelle à partir de ces fichiers.

Pour appliquer la règle de sélection à de nouveaux essais, passer leurs dossiers avec
`--candidates`. Pour évaluer les modèles conservés et comparer les trois pilotes dans la même fenêtre 3D :

```powershell
.\.venv\Scripts\python.exe -m rocket_hugo.evaluate_phase0 --algo ppo --model players/hugo/models/phase0-ppo-residual.zip --split official
.\.venv\Scripts\python.exe -m rocket_hugo.evaluate_phase0 --algo sac --model players/hugo/models/phase0-sac-residual.zip --split official
.\.venv\Scripts\python.exe -m rocket_hugo.demo_3d --mode compare --seed 10046
```

Dans la fenêtre : `P` reprend le vol, `1` choisit le pilote seul, `2` PPO et `3` SAC.
`D` change le départ et `V` / `B` permettent de comparer des versions d'entraînement
enregistrées. Le tableau conserve l'issue, la distance finale et la vitesse au premier contact
sur les vols terminés avec le départ et la version sélectionnés. Voir le
[README Hugo](../README.md) pour recréer les sauvegardes intermédiaires.

Les rapports locaux font foi pour cette série. WandB n'a pas produit de journal exploitable :
Windows a bloqué le lancement de son service, même en mode hors ligne.
