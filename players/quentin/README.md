# Quentin

Environnement et agents de Quentin. Voir le [contrat](../../CONTRACT.md) pour le problème et les critères.

## Où j'en suis

| Phase | Validée | Taux de réussite | Précision (médiane) | Vitesse au contact | Algorithme |
|---|---|---|---|---|---|
| 0 | Oui | 96/100 | 0,21 m | 0,85 m/s | Imitation supervisée, DAgger |

Pilote programmé à la main, utilisé comme référence en phase 0 sur les 100
graines 10 000 à 10 099. Il a réussi les 100 atterrissages, avec un écart
médian de 0,16 m au centre, une vitesse verticale moyenne de 0,77 m/s vers
le bas et une inclinaison moyenne de 0,65° au premier contact.

## Lancer

### Préentraîner SAC par imitation

Dans les essais précédents, SAC devait découvrir par exploration des poses
réussies à des hauteurs bien inférieures aux 80 à 120 m de la phase 0.
Le pilote de référence sait déjà effectuer un vol complet. On enregistre ses
observations et ses commandes pour donner au réseau une politique de départ.
Cette approche s'appelle le *behavior cloning*. En robotique, des travaux ont
aussi utilisé des démonstrations pour faciliter l'exploration et préentraîner
une politique avant de l'affiner par renforcement. [Nair et al., ICRA 2018](https://arxiv.org/abs/1709.10089)
[Rajeswaran et al., RSS 2018](https://arxiv.org/abs/1709.10087)

1. **Collecte.** On joue 256 départs complets sur les graines 2000 à 2255.
   Le pilote réussit 255 vols. On utilise les 67 054 décisions de ces vols
   réussis pour apprendre à prédire ses trois commandes par erreur quadratique.
2. **Affinage.** On initialise l'acteur SAC avec les poids appris et son
   tampon de rejeu avec les 67 347 transitions de collecte, y compris celles
   du vol raté. SAC joue ensuite ses propres épisodes à difficulté complète.
   Il ajuste sa politique à partir des récompenses `progress`.
3. **Mesure.** Les graines 6000 à 6019 servent à sélectionner le checkpoint
   d'imitation au meilleur taux de pose, avec priorité au premier en cas
   d'égalité, puis à suivre SAC. Les graines 6100 à 6139 servent à comparer
   ce checkpoint à la fin de l'affinage. Les graines officielles 10 000 à
   10 099 sont réservées à l'évaluation officielle.

Le pilote transmet ses commandes pendant la collecte. En évaluation,
le réseau reçoit les 19 valeurs d'observation et produit seul les trois
commandes en mode déterministe. Le scénario, la physique et le critère
de réussite suivent le [contrat commun](../../CONTRACT.md).

Depuis la racine du dépôt, collecte une seule fois, puis entraîne les trois
modèles. Si le fichier de démonstrations existe déjà, réutilise-le.

```powershell
uv run --locked python -m rocket_quentin.demonstrations
foreach ($trainingSeed in 0, 1, 2) {
    uv run --locked python -m rocket_quentin.train_imitation --epochs 30 --sac-steps 6000 --seed $trainingSeed
}
```

Chaque run utilise le CPU et deux couches de 128 neurones. L'imitation
utilise des lots de 256 et un taux d'apprentissage de 0,001. L'affinage
utilise des lots de 128, un taux de 0,0001, un facteur d'actualisation de
0,995 et le réglage automatique de l'entropie de SAC.
Avec `--sac-steps 0`, on s'arrête après l'imitation. Le dossier du run
contient `config.json`, `report.json`, le modèle d'imitation dans `imitation/`
et, si l'affinage est activé, les checkpoints dans `checkpoints/` et le modèle
de fin de run dans `sac/`.

### Affiner SAC en limitant l'écart à la politique initiale

Dans cette variante, chaque mise à jour de l'acteur SAC ajoute à sa perte
une pénalité égale à 100 fois l'erreur quadratique entre ses commandes
actuelles et celles du réseau d'imitation sauvegardé. On mesure cet écart
sur un lot de 256 observations tiré parmi les démonstrations réussies.
SAC conserve sa récompense et ajuste toujours sa politique sur ses
propres expériences et les transitions de collecte. En évaluation, seul
l'acteur final produit les commandes.

Pour reprendre exactement les mêmes poids qu'un essai précédent, remplace
`imitation-s0-EXEMPLE` par le nom du run d'origine et utilise sa graine.

```powershell
uv run --locked python -m rocket_quentin.train_imitation --imitation-dir "players/quentin/runs/imitation-s0-EXEMPLE/imitation" --seed 0 --sac-steps 6000 --sac-anchor 100
```

Les trois variantes sont évaluées en déterministe sur les mêmes 40 départs de
comparaison. À titre de référence, le pilote programmé y pose la fusée 40
fois. Les deux colonnes SAC donnent le score en fin de budget.

| Graine d'entraînement | Imitation seule | SAC à 6 000 décisions | SAC avec pénalité à 6 000 décisions |
|---:|---:|---:|---:|
| 0 | 34/40 | 0/40 | 35/40 |
| 1 | 33/40 | 0/40 | 28/40 |
| 2 | 30/40 | 0/40 | 26/40 |
| Total | 97/120 | 0/120 | 89/120 |

L'imitation seule produit 12 contacts trop rapides, 9 sorties de zone liées
à l'inclinaison et 2 chutes à l'eau. Avec SAC sans pénalité, les 120 vols de
comparaison finissent hors zone. La pénalité préserve une partie du pilotage
appris. En fin de budget, son effet par rapport à l'imitation varie selon la
graine et le total de réussites baisse. Les expériences de curriculum ci-dessous
utilisent d'autres départs et budgets. Elles donnent le contexte du choix de
l'imitation. Le tableau ci-dessus compare l'effet de l'affinage à partir des
mêmes poids.

Pour le test officiel, le candidat est choisi parmi les checkpoints SAC avec
pénalité selon le nombre de poses en validation, avec priorité au checkpoint
le plus tôt en cas d'égalité, puis à la plus petite graine. Cette règle a
retenu la graine 0 à 4 000 décisions avec 20/20 en validation. Ce checkpoint
obtient 38/40 en comparaison, avec une sortie de zone et un crash.

Sur les 100 graines officielles, il obtient **91 poses**, pour un seuil de
95. La distance médiane au centre est de 0,81 m. La vitesse verticale
moyenne au premier contact est de 0,67 m/s et l'inclinaison moyenne de 1,51°.
Les neuf échecs sont cinq sorties de zone liées à l'inclinaison, deux
atterrissages durs et deux chutes à l'eau. Cet essai SAC reste sous le
seuil de la phase 0.

Pour rejouer ce checkpoint local, utilise la commande ci-dessous. Après un
nouvel entraînement, remplace `imitation-s0-xk5eiozj` par le dossier affiché
par le run avec `--seed 0 --sac-anchor 100`.

```powershell
uv run --locked python -m rocket_quentin.evaluate --run-dir "players/quentin/runs/imitation-s0-xk5eiozj/checkpoints/sac-04000"
```

### Comparer le coefficient d'entropie initial de SAC

On reprend les trois checkpoints d'imitation des essais précédents. Les deux
conditions utilisent la même pénalité de poids 100, le même tampon de rejeu,
la même récompense `progress` et 6 000 décisions par graine. Le coefficient
d'entropie démarre à 1 dans le contrôle et à 0,1 dans la variante. SAC continue
de l'ajuster automatiquement, avec la même entropie cible dans les deux cas.

On suit les checkpoints à 2 000, 4 000 et 6 000 décisions sur les graines
de validation 6000 à 6019. On compare les modèles finaux et l'imitation de
départ sur 40 nouveaux départs, avec les graines 6200 à 6239. Le critère
retenu est d'améliorer l'imitation à la fin sur au moins deux graines
d'entraînement sur trois. Pour ces graines, le score final de validation
doit être à deux poses ou moins du meilleur checkpoint du run.

Depuis la racine du dépôt, remplace `$imitationRun` par le chemin du run
d'imitation à reprendre et utilise sa graine pour `$trainingSeed`.

```powershell
foreach ($initialEntropy in @("1", "0.1")) {
    uv run --locked python -m rocket_quentin.train_imitation --imitation-dir "$imitationRun/imitation" --seed $trainingSeed --sac-steps 6000 --sac-anchor 100 --sac-initial-entropy $initialEntropy --entropy-comparison
}
```

Pour comparer un checkpoint intermédiaire, passe son dossier à
`rocket_quentin.evaluate --entropy-comparison --run-dir`.

| Graine | Imitation seule | SAC à 6 000, coef. 1 | SAC à 6 000, coef. 0,1 | Validation coef. 1 (2k / 4k / 6k) | Validation coef. 0,1 (2k / 4k / 6k) |
|---:|---:|---:|---:|---:|---:|
| 0 | 34/40 | 36/40 | 29/40 | 18 / 20 / 18 sur 20 | 18 / 17 / 15 sur 20 |
| 1 | 38/40 | 33/40 | 30/40 | 16 / 18 / 15 sur 20 | 17 / 16 / 14 sur 20 |
| 2 | 28/40 | 29/40 | 30/40 | 18 / 18 / 19 sur 20 | 19 / 19 / 17 sur 20 |
| Total | 100/120 | 98/120 | 89/120 | — | — |

Le coefficient 0,1 améliore l'imitation sur une seule graine. Les trois
scores finaux de validation baissent par rapport à leur meilleur checkpoint.
Le réglage 0,1 échoue au critère de régularité. Après 2 000 décisions, sur
un échantillon des observations de démonstration, ses actions déterministes
s'écartent en moyenne d'environ 0,007 des actions imitées, contre 0,020
avec le coefficient 1, sur l'échelle des commandes de −1 à +1.
Le checkpoint de référence
au coefficient 1 et à 4 000 décisions obtient 38/40 sur ces nouveaux
départs. Aucun nouveau modèle de cet essai n'a été testé sur les graines
officielles.

### Apprendre à corriger ses propres trajectoires avec DAgger

Les démonstrations initiales contiennent surtout les situations rencontrées
par le pilote de référence. Avec DAgger, le réseau commande la fusée et le
pilote indique en parallèle quelle commande il aurait choisie dans chaque
situation. On ajoute ces paires observation–commande aux démonstrations
pour entraîner le réseau aussi sur les écarts qu'il crée lui-même.
[Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)

Pour chaque modèle d'imitation, on fait deux tours de 32 départs complets
(graines 3000 à 3063) et 10 époques supervisées par tour. Les exemples
des deux tours s'accumulent avec les démonstrations réussies d'origine.
On choisit le checkpoint sur les graines 6000 à 6019, avec priorité au
premier en cas d'égalité. On compare ce checkpoint à l'imitation seule
sur les 40 départs 6300 à 6339.

```powershell
uv run --locked python -m rocket_quentin.train_dagger --imitation-dir "$imitationRun/imitation"
```

| Graine | Imitation seule | Après DAgger | Corrections ajoutées |
|---:|---:|---:|---:|
| 0 | 34/40 | 38/40 | 17 574 |
| 1 | 36/40 | 38/40 | 18 785 |
| 2 | 36/40 | 38/40 | 18 518 |
| Total | 106/120 | 114/120 | 54 877 |

Sur les mêmes départs, les atterrissages durs passent de huit à quatre et
les sorties de zone de quatre à deux. Les trois graines progressent.
Cette comparaison mesure l'ensemble de l'étape DAgger, qui ajoute des
corrections et de l'entraînement supervisé.
Pour l'évaluation officielle, on retient la graine 1. Son checkpoint est
le premier à obtenir 20/20 en validation, à la cinquième époque du
deuxième tour. Ce modèle d'imitation supervisée obtient **96/100**, au-dessus
du seuil de phase 0 de 95 poses. La distance médiane au centre est de
0,21 m. Au premier contact, la vitesse verticale moyenne est de 0,85 m/s
et l'inclinaison moyenne de 1,42°. Les quatre échecs sont trois sorties
de zone liées à l'inclinaison et un atterrissage dur.

Pour reproduire les 100 vols officiels avec le modèle sauvegardé et son
fichier de normalisation, lance cette commande depuis la racine du dépôt.

```powershell
uv run --locked python -m rocket_quentin.evaluate --model players/quentin/models/phase0-dagger.zip
```

### Entraîner SAC avec le curriculum

Depuis la racine du dépôt, lance les deux versions de la récompense avec
les mêmes graines d'entraînement et le même budget.

```powershell
foreach ($rewardMode in @("sparse", "progress")) {
    foreach ($trainingSeed in 0, 1) {
        uv run --locked python -m rocket_quentin.train_sac --reward $rewardMode --steps 6000 --eval-every 1000 --seed $trainingSeed
    }
}
```

Au niveau 0, la fusée part de 4 à 9 cm au-dessus du pont, près du centre et
presque droite. SAC contrôle les trois actions dès le départ. Tous les 1 000
pas dans cet exemple, le script évalue la politique sans exploration sur
20 graines de développement.
À partir de 16 réussites sur 20, il augmente la difficulté de 0,1 pour les
prochains départs. Au niveau 1, les départs suivent le contrat complet.

Chaque run enregistre sa configuration, ses résultats par niveau, le modèle
et les statistiques de normalisation dans `players/quentin/runs/`. W&B
enregistre le run hors ligne par défaut.

- `sparse` utilise les récompenses de base. Atterrissage réussi +100, échec
  physique −100, vol et limite de temps 0.
- `progress` ajoute `0,995 × potentiel_après − potentiel_avant`. Le potentiel
  vaut `−10 × (h/(h+1) + d/(d+2))`, avec `h` la hauteur au-dessus du pont,
  ramenée à zéro si elle devient négative, et `d` la distance horizontale à
  son centre, en mètres. Le potentiel final vaut zéro pour une fin physique.
  À la limite de temps, on garde le potentiel du dernier état.

Pour comparer un modèle sur les niveaux 0, 0,1 et 0,2, ajoute
`--development` à la commande d'évaluation. Les graines 5100 à 5119 sont
distinctes de celles qui décident les promotions.

Comparaison après 6 000 décisions par run. Chaque cellule donne le nombre
d'atterrissages sur les mêmes 20 départs de comparaison.

| Récompense | Graine d'entraînement | Difficulté 0 | Difficulté 0,1 | Difficulté 0,2 |
|---|---:|---:|---:|---:|
| `sparse` | 0 | 15/20 | 11/20 | 0/20 |
| `progress` | 0 | 20/20 | 18/20 | 3/20 |
| `sparse` | 1 | 7/20 | 9/20 | 0/20 |
| `progress` | 1 | 14/20 | 13/20 | 4/20 |

Les rapports `comparison.json` sont dans les dossiers des quatre runs.

Premiers essais à 6 000 décisions par graine, sur les graines d'entraînement 0
et 1. Le départ initial à 30–65 cm a donné 0 réussite sur 20 pour les
deux runs. Avec le départ à 4–9 cm, la graine 0 a atteint 17/20 au niveau
0 après 2 000 décisions, puis 10/20 au niveau 0,1 à la fin. La graine 1 a
terminé à 9/20 au niveau 0. Ces mesures utilisent les départs de développement.

### Comparer le palier intermédiaire à 0,15

Les deux curricula utilisent la récompense `progress`. Le curriculum
`standard` passe de 0,1 à 0,2. Le curriculum `intermediate` ajoute 0,15
entre les deux. Le seuil de promotion reste 16 poses sur 20.

```powershell
foreach ($curriculumMode in @("standard", "intermediate")) {
    foreach ($trainingSeed in 0, 1, 2) {
        uv run --locked python -m rocket_quentin.train_sac --reward progress --curriculum $curriculumMode --steps 20000 --eval-every 1000 --seed $trainingSeed
    }
}
```

Pour comparer les modèles au palier 0,2, rejoue les mêmes 20 départs de
développement pour chaque dossier de run. Le rapport compte notamment
les atterrissages durs et les sorties avec une inclinaison supérieure à 60°.

```powershell
uv run --locked python -m rocket_quentin.evaluate --development-level 0.2 --run-dir "players/quentin/runs/sac-progress-intermediate-s0-EXEMPLE"
```

Résultats après 20 000 décisions d'entraînement par run, sur les graines
d'évaluation 5100 à 5119 au palier 0,2.

| Graine d'entraînement | Poses standard | Poses avec 0,15 | Sorties par inclinaison standard | Sorties par inclinaison avec 0,15 |
|---:|---:|---:|---:|---:|
| 0 | 1/20 | 20/20 | 9 | 0 |
| 1 | 14/20 | 10/20 | 1 | 6 |
| 2 | 4/20 | 4/20 | 7 | 6 |
| Total | 19/60 | 34/60 | 17 | 12 |

Aucun atterrissage dur dans les six évaluations finales à 0,2. Le gain
global vient de la graine 0. Le palier 0,15 n'améliore pas toutes les graines.
Les rapports détaillés sont dans `comparison-level-0.2.json` dans chaque run.

### Isoler la difficulté liée à l'altitude

Pour cette expérience, `--altitude-only` augmente seulement la hauteur de
départ. Les autres variables restent tirées dans leurs plages du palier 0.
On garde SAC, la récompense `progress`, le curriculum `intermediate` et
20 000 décisions par graine, comme dans le comparatif précédent.

```powershell
foreach ($trainingSeed in 0, 1, 2) {
    uv run --locked python -m rocket_quentin.train_sac --reward progress --curriculum intermediate --altitude-only --probe-every 4000 --steps 20000 --eval-every 1000 --seed $trainingSeed
}
```

Tous les 4 000 pas, le même modèle est évalué aux paliers 0,2 et 0,3, sur les
graines 5100 à 5119. Les résultats de `fixed-levels.json` servent au suivi.
Les promotions restent décidées sur les graines 5000 à 5019. Chaque
checkpoint contient le modèle et les statistiques de normalisation associées.

Pour rejouer un checkpoint à 0,3, ajoute `--altitude-only` pour isoler la
hauteur. Sans ce drapeau, le test fait aussi varier les autres conditions de
départ comme dans le curriculum précédent.

```powershell
uv run --locked python -m rocket_quentin.evaluate --development-level 0.3 --altitude-only --run-dir "players/quentin/runs/sac-progress-intermediate-altitude-s0-EXEMPLE/checkpoints/step-08000"
```

Résultats des modèles entraînés en faisant varier uniquement l'altitude, après
20 000 décisions. Chaque colonne utilise les mêmes 20 graines par modèle.

| Graine | Altitude seule à 0,2 | Altitude seule à 0,3 | Paramètres combinés à 0,2 | Paramètres combinés à 0,3 |
|---:|---:|---:|---:|---:|
| 0 | 2/20 | 0/20 | 1/20 | 0/20 |
| 1 | 10/20 | 3/20 | 10/20 | 3/20 |
| 2 | 4/20 | 3/20 | 4/20 | 3/20 |
| Total | 16/60 | 6/60 | 15/60 | 6/60 |

Les trois modèles du comparatif précédent, entraînés en augmentant tous les
paramètres, obtenaient 34/60 à 0,2 sur les départs combinés. Au palier 0,3,
ils obtiennent 5/60 sur chacun des deux types de départ. Dans l'expérience
altitude seule, la graine 2 passe de 12/20 à 0,2 à 12 000 décisions à 4/20
à la fin, sur les mêmes départs de comparaison.

Les rapports `comparison-combined.json` et `comparison-altitude-experiment.json`
contiennent les résultats détaillés des nouveaux runs et des runs de référence.

### Garder des départs du palier précédent

Le paramètre `--previous-level-share 0.25` donne à chaque nouvel épisode
après une promotion 25 % de chances de partir au palier précédent et 75 %
de partir au nouveau. La valeur `0` utilise seulement le palier courant.
La promotion continue à dépendre de 16 réussites sur 20 au palier courant.

```powershell
foreach ($previousShare in @("0", "0.25")) {
    foreach ($trainingSeed in 0, 1, 2) {
        uv run --locked python -m rocket_quentin.train_sac --reward progress --curriculum intermediate --previous-level-share $previousShare --probe-every 4000 --steps 20000 --eval-every 1000 --seed $trainingSeed
    }
}
```

Chaque checkpoint mesure séparément les paliers courant et précédent, ainsi
que les niveaux fixes 0,2 et 0,3, sur les mêmes 20 départs de comparaison par
niveau. Aucun mélange n'est appliqué pendant ces tests. Les résultats sont
dans `fixed-levels.json` et les checkpoints dans `checkpoints/`.

Résultats après 20 000 décisions, sur les graines de comparaison 5100 à 5119.

| Graine | Palier 0,2 sans mélange | Palier 0,2 avec 25 % | Palier 0,3 sans mélange | Palier 0,3 avec 25 % |
|---:|---:|---:|---:|---:|
| 0 | 20/20 | 10/20 | 1/20 | 1/20 |
| 1 | 10/20 | 12/20 | 1/20 | 0/20 |
| 2 | 4/20 | 10/20 | 3/20 | 3/20 |
| Total | 34/60 | 32/60 | 5/60 | 4/60 |

Avec le mélange et la graine 0, le score au palier précédent 0,15 passe
de 18/20 à 12 000 décisions à 5/20 à 16 000 décisions, puis 10/20 à la fin.
Ces trois mesures utilisent les mêmes départs.

### Compter les réussites réellement collectées pendant l'entraînement

Chaque run enregistre maintenant `training-outcomes.json`. À chaque intervalle
d'évaluation, ce fichier compte les issues des épisodes d'entraînement
terminés dans l'intervalle, ventilées par difficulté effectivement tirée au
départ de chaque épisode. Il attribue les décisions d'un épisode à
l'intervalle où cet épisode se termine. Les décisions des épisodes en cours
ne sont donc pas comptées dans les totaux par issue. Le fichier indique aussi
la taille du buffer SAC, le coefficient d'entropie et sa cible.

La température SAC est réglée comme dans les valeurs par défaut de
Stable-Baselines3 2.9.2. `ent_coef="auto"` part de 1 et s'ajuste pendant
l'apprentissage. `target_entropy="auto"` vaut −3 ici, car la fusée a trois
commandes continues. Les autres réglages sont dans `config.json`.

Trois répétitions du curriculum mélangé, à 20 000 décisions, ont reproduit
les scores d'évaluation précédents avec la télémétrie activée. Réussites
parmi les épisodes réellement terminés pendant l'entraînement :

| Graine | Palier 0 | Palier 0,1 | Palier 0,15 | Palier 0,2 | Palier 0,3 | Coefficient d'entropie final |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11/38 | 0/2 | 8/17 | 12/34 | — | 0,0264 |
| 1 | 35/64 | 51/51 | 32/68 | 21/50 | 0/2 | 0,0791 |
| 2 | 83/166 | 34/47 | — | — | — | 0,0563 |

Pour la graine 0, entre 12 000 et 16 000 décisions, SAC a réussi 4 des
6 épisodes terminés partis du palier 0,15 et 2 des 12 partis de 0,2.
Pendant cette fenêtre, son score déterministe sur les départs de
comparaison à 0,15 est descendu de 18/20 à 5/20.

### Entraîner plus longtemps avec un plafond à 0,2

L'option `--max-difficulty 0.2` conserve les paliers intermédiaires jusqu'à
0,2 et empêche toute promotion au-delà. On garde la récompense `progress`
et 25 % de départs au palier précédent. Le budget est de 32 000 décisions
par graine.

```powershell
foreach ($trainingSeed in 0, 1, 2) {
    uv run --locked python -m rocket_quentin.train_sac --reward progress --curriculum intermediate --previous-level-share 0.25 --max-difficulty 0.2 --probe-levels 0.15 0.2 --probe-every 4000 --steps 32000 --eval-every 1000 --seed $trainingSeed
}
```

Sur les départs fixes de comparaison à 0,2, le critère choisi pour cette
expérience est d'obtenir au moins 16 poses sur 20 à trois checkpoints
consécutifs. On suit en parallèle les poses à 0,15 pour mesurer la
conservation des performances au palier précédent.

Résultats sur les mêmes 20 graines de comparaison par palier à chaque
checkpoint. Les colonnes des 20 000 à 32 000 décisions portent sur le
palier 0,2.

| Graine | Arrivée à 0,2 | 20 000 | 24 000 | 28 000 | 32 000 | Palier 0,15 à 32 000 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 000 | 9/20 | 14/20 | 11/20 | 20/20 | 20/20 |
| 1 | 7 000 | 12/20 | 20/20 | 12/20 | 20/20 | 20/20 |
| 2 | 22 000 | 12/20 | 12/20 | 11/20 | 10/20 | 12/20 |

Les trois graines ont atteint 0,2. Elles y ont réussi respectivement
57/98, 90/127 et 23/41 épisodes d'entraînement terminés. Le score de
comparaison passe de 33/60 à 20 000 décisions à 50/60 à 32 000. Aucun
run ne réussit 16/20 à trois checkpoints consécutifs. L'expérience mesure
l'évolution dans les runs plafonnés. Elle ne comporte pas de run sans
plafond au même budget de 32 000 décisions.

### Évaluer sur les départs du contrat

Pour mesurer le pilote de référence ou un modèle sauvegardé sur les 100
graines officielles, utilise les commandes ci-dessous. Remplace le dossier
d'exemple par le chemin affiché au début de ton run.

```powershell
uv run --locked python -m rocket_quentin.evaluate
uv run --locked python -m rocket_quentin.evaluate --run-dir "players/quentin/runs/sac-progress-s0-EXEMPLE"
```

### Essayer un épisode Gymnasium

Depuis la racine du dépôt, colle ce bloc dans PowerShell. La commande
`[-1, 0, 0]` coupe le moteur et garde la tuyère dans l'axe.

```powershell
@'
from rocket_quentin.env import RocketLandingEnv

env = RocketLandingEnv()
observation, info = env.reset(seed=12)
print(f"Altitude de départ : {observation[2]:.1f} m")
while True:
    observation, reward, terminated, truncated, info = env.step([-1, 0, 0])
    if terminated or truncated:
        print(f"Fin après {env.data.time:.2f} s : {info}, récompense {reward:+g}")
        break
env.close()
'@ | uv run --locked python -
```

Les trois valeurs de l'action vont de −1 à +1. La première règle la poussée
de 0 à 19 620 N. Les deux autres règlent chacune un angle de −10° à +10°.
Une commande reste active pendant dix pas MuJoCo de 5 ms, sauf si l'épisode
se termine plus tôt.

L'observation contient 19 valeurs dans cet ordre :

- position horizontale par rapport à la barge, puis hauteur du pied le plus bas au-dessus du pont ;
- vitesse linéaire et vitesse angulaire, chacune sur trois axes ;
- directions des axes x et z de la fusée, exprimées dans le repère monde ;
- quatre indicateurs de contact avec le dessus du pont.

La première récompense est simple : +100 à l'atterrissage, −100 en cas
d'échec physique, 0 en vol et à la limite de temps.

### Voir la scène et la poussée

Pour voir **le même épisode Gymnasium** en 3D, avec la graine 12 et le moteur
coupé, lance cette commande depuis la racine du dépôt :

```powershell
uv run --locked python -m rocket_quentin.demo_episode
```

La caméra suit la descente et la fenêtre reste ouverte sur la dernière image.
Appuie sur `R` dans la fenêtre pour rejouer le même départ, ou ajoute
`--seed 13` à la commande pour essayer un autre départ.

### Comparer le pilote vertical au moteur coupé

Ces deux commandes partent du même scénario de développement. La fusée est
droite, à 100 m au-dessus du centre du pont, et descend à 10 m/s.

```powershell
uv run --locked python -m rocket_quentin.demo_episode --mode vertical
uv run --locked python -m rocket_quentin.demo_episode --mode vertical-freefall
```

Le pilote vise une vitesse de descente de plus en plus faible à l'approche du
pont. Si la fusée descend trop vite, il augmente la poussée. Il coupe le moteur
au premier contact. La flèche jaune indique la poussée. À la fin, PowerShell
affiche la vitesse au contact pour comparer les deux essais.

### Comparer le redressement de la fusée

Ces deux essais démarrent au-dessus du centre à 100 m avec la même inclinaison
de 5°. Les deux utilisent le freinage vertical. Dans le premier, un correcteur
PD oriente aussi la tuyère pour redresser la fusée. Il corrige l'inclinaison
et freine la rotation pour éviter de dépasser la verticale.

```powershell
uv run --locked python -m rocket_quentin.demo_episode --mode attitude
uv run --locked python -m rocket_quentin.demo_episode --mode attitude-uncorrected
```

### Comparer le guidage vers le centre du pont

Les deux essais partent à 100 m d'altitude, à environ 14,4 m du centre,
avec une vitesse horizontale de 1 m/s qui éloigne la fusée du pont.
Le pilote latéral incline la fusée pour corriger le décalage et la vitesse.
Le pilote précédent stabilise seulement l'orientation et la descente.

```powershell
uv run --locked python -m rocket_quentin.demo_episode --mode lateral
uv run --locked python -m rocket_quentin.demo_episode --mode lateral-uncorrected
```

Pour rejouer un départ aléatoire de la phase 0 avec le pilote complet :

```powershell
uv run --locked python -m rocket_quentin.demo_episode --mode pilot --seed 12
```

### Voir la scène et tester le moteur

Depuis la racine du dépôt, ouvre la scène MuJoCo de la phase 0 :

```powershell
uv run --locked python -m mujoco.viewer --mjcf=players/quentin/src/rocket_quentin/assets/scene.xml
```

La fusée démarre à 15 cm du pont pour observer la gravité et les contacts. Ce départ
sert à tester la scène. Les départs du contrat seront générés à chaque épisode par
l'environnement Gymnasium.

Pour voir la poussée en action, lance la démo interactive :

```powershell
uv run --locked python -m rocket_quentin.demo_thrust
```

Dans la fenêtre, appuie sur `1` pour couper le moteur, `2` pour le vol
stationnaire, `3` pour la pleine poussée ou `4` pour incliner la tuyère.
Chaque touche relance un essai de 1,4 seconde. La flèche jaune sous la tuyère
indique la direction de la poussée, et sa longueur représente l'intensité.

Pour vérifier la poussée et les limites de la tuyère :

```powershell
uv run --locked pytest players/quentin/tests/test_thrust.py -v
```

Dans la simulation, `apply_thrust` prend une force en newtons et deux angles en
degrés. On l'appelle avant chaque pas de physique. Les deux angles à zéro
orientent la poussée suivant l'axe de la fusée. Droite et au repos, la fusée
reste à la même hauteur avec une poussée de 9 810 N, qui compense son poids.

```bash
# entraînement :
# évaluation (graines 10 000 à 10 099) :
```

## Notes

Choix de conception, essais, ce qui a marché ou pas.
