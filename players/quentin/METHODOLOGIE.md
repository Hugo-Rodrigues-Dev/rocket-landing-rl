# Comment nous avons entraîné l’agent d’atterrissage

Le projet poursuit l’apprentissage par renforcement avec **SAC à partir de poids aléatoires**, sans démonstrations ni actions du pilote. Le simulateur couvre les cinq phases du contrat. Les modèles DAgger servent de comparaison et leur entraînement relève de l’imitation supervisée. [Commandes SAC](README.md#apprentissage-par-renforcement-avec-sac)

Dans l’historique de phase 0, le meilleur essai SAC avec préentraînement par imitation a réussi **91 poses sur 100**, pour un seuil de 95. Les modèles d’imitation ont atteint les seuils du contrat pour les phases 0 à 4. La phase 1 réutilise le modèle DAgger de phase 0.

## Comprendre DAgger avec un exemple

Imagine un vol où la fusée arrive à deux mètres du pont en descendant trop vite.

1. **Le pilote de référence montre un vol.** C’est un programme écrit à la main qui lit la position, la vitesse, les angles et les autres mesures disponibles. On enregistre la situation à chaque pas et la commande qu’il donne. Le réseau apprend à prédire ces commandes. Cet apprentissage sur des exemples s’appelle le *behavior cloning*, ou imitation supervisée.
2. **Le réseau fait son propre vol.** Il commet parfois une petite erreur, qui change la suite de sa trajectoire. S’il arrive trop vite à deux mètres du pont, on demande au pilote quelle commande il choisirait dans cette situation précise. Le réseau garde les commandes pendant le vol. On enregistre la réponse du pilote et on l’ajoute aux exemples pour le prochain entraînement.
3. **Le réseau passe le test.** À chaque pas, il lit les mesures courantes et calcule lui-même sa commande. Ses poids sont fixés pour toute l’évaluation.

**DAgger** signifie *Dataset Aggregation*, ou agrégation de données. C’est la répétition de l’étape 2. En pratique, une erreur de pilotage change les observations suivantes. En faisant voler le réseau lui-même, on recueille des exemples de correction dans les situations où il risque de se retrouver. [Article original de Ross, Gordon et Bagnell](https://proceedings.mlr.press/v15/ross11a.html)

Le pilote explique donc **d’où vient le comportement appris**. Le réseau explique **qui prend les décisions en vol**.

DAgger est bien du **machine learning**. Les poids du réseau sont ajustés sur des exemples, comme dans tout apprentissage supervisé. Le simulateur fait évoluer la fusée en fonction des actions du réseau et produit ainsi les situations où il doit apprendre à se corriger. Le pilote de référence fournit ensuite la commande à apprendre pour chacune de ces situations.

En DAgger, quand la fusée descend trop vite, l’écart crée un exemple d’entraînement et le pilote indique comment freiner. En apprentissage par renforcement, la pose réussie ou ratée fournit une récompense et l’algorithme doit estimer quelles actions ont contribué au résultat. **Notre modèle d’imitation apprend à corriger ses erreurs à partir des réponses du pilote.** C’est avec SAC que nous avons testé l’apprentissage à partir des récompenses.

## Où intervient SAC dans notre historique

**SAC**, pour *Soft Actor-Critic*, est une méthode d’apprentissage par renforcement. L’acteur choisit des actions et les critiques estiment leur intérêt à partir des récompenses observées. L’objectif de l’acteur combine la récompense attendue avec un terme d’entropie qui favorise la diversité des actions pendant l’entraînement. [Article original de Haarnoja et al.](https://proceedings.mlr.press/v80/haarnoja18b.html)

| Essai | Comment les poids étaient appris | Ce qu’on a observé |
|---|---|---|
| SAC avec curriculum | Le réseau apprend par récompense. Il commence près du pont, puis la difficulté augmente quand il réussit suffisamment de poses. | Les résultats varient selon les graines. Au palier 0,2, aucun des trois runs plafonnés à 32 000 décisions ne maintient au moins 16 poses sur 20 à trois checkpoints consécutifs. |
| Imitation seule en phase 0 | Le réseau apprend les commandes du pilote sur des vols complets. | 34, 33 et 30 poses sur 40 selon la graine. |
| Affinage SAC après imitation | SAC reprend l’acteur imité et utilise les récompenses. Son tampon de rejeu contient aussi des transitions des démonstrations. | Après 6 000 décisions, les trois modèles de cet essai donnent 0/40. |
| Affinage SAC avec pénalité d’écart | SAC optimise toujours la récompense. Une pénalité de poids 100 limite l’écart entre ses commandes et celles du réseau au début de l’affinage. | Le checkpoint retenu à 4 000 décisions réussit 38/40 en comparaison, puis **91/100** au test officiel de phase 0. |
| Imitation puis DAgger | Le réseau apprend sur des vols du pilote, puis sur les corrections du pilote dans les situations générées par le réseau. | Sur un autre jeu de 40 départs, les trois modèles atteignent chacun 38/40. Le modèle retenu obtient **96/100** au test officiel de phase 0. |

Les comparaisons ont leurs propres graines et budgets. Les chiffres se lisent avec le protocole de chaque essai. Le tableau complet des expériences se trouve dans le [README](README.md).

Les essais SAC avec curriculum utilisent des couches de 64 et 64 neurones, un taux d’apprentissage de 0,0003, des lots de 128, un tampon de 100 000 transitions et un facteur d’actualisation de 0,995. L’affinage après imitation utilise deux couches de 128 neurones et un taux de 0,0001. Nous avons aussi comparé un coefficient d’entropie initial de 1 à 0,1. Sur les mêmes départs de comparaison, après 6 000 décisions et avec la pénalité, les deux configurations totalisent respectivement 98 et 89 poses sur 120. [Paramètres du curriculum](src/rocket_quentin/sac/train_sac.py) · [Affinage après imitation](src/rocket_quentin/dagger/train_imitation.py)

### Pourquoi le modèle de phase 0 est dans un fichier SAC

Pour la phase 0, nous réutilisons **l’acteur de Stable-Baselines3** et son format de sauvegarde. Le script DAgger charge cet acteur et met à jour ses poids avec une erreur supervisée sur les commandes du pilote. Le type de fichier indique comment charger le réseau. Le script d’entraînement indique comment ses poids ont été obtenus. Les phases 2 à 4 sauvegardent leurs réseaux PyTorch dans des fichiers `.pt`. [Entraînement DAgger de phase 0](src/rocket_quentin/dagger/train_dagger.py) · [Réseaux des phases suivantes](src/rocket_quentin/dagger/learned_landing.py)

## Récompense du simulateur et perte d’entraînement

**La récompense de base de Gymnasium** vaut +100 pour une pose, −100 pour un échec physique et 0 en vol ou à la limite de temps. Dans nos essais SAC, la variante `progress` ajoute un terme de progression. Un potentiel lié à la hauteur et à l’écart horizontal augmente lorsque la fusée se rapproche du pont et de son centre. Le terme ajouté est `0,995 × potentiel suivant − potentiel courant`. [Environnement](src/rocket_quentin/common/env.py) · [Récompense de progression](src/rocket_quentin/common/reward.py)

**La perte supervisée du modèle d’imitation** mesure l’écart entre les sorties du réseau et les commandes de référence. Elle ajoute une perte de classification pour le choix moteur allumé ou coupé, avec un poids de 0,075. Les exemples sous 3 m comptent double. Les mises à jour de ce réseau utilisent ces exemples, et la sélection de checkpoints utilise les atterrissages réussis et leur précision sur des vols de validation. [Code de l’entraînement](src/rocket_quentin/dagger/train_landing.py)

## Ce que le réseau de phase 4 a appris

Le pilote de référence adapte sa trajectoire au vent et utilise le RCS pour amortir les rotations. Il fournit des exemples de pilotage jusqu’au toucher. Le réseau de phase 4 reçoit les **31 mesures** de l’environnement, dont la position et la vitesse relatives au pont, son orientation, le carburant, les contacts et le vent actuel. On ajoute une transformation logarithmique de l’altitude puis on normalise les entrées. Le réseau a **trois couches de 128 neurones** avec activation SiLU. Il décide toutes les **50 ms** quand allumer le moteur et commande la poussée, les deux angles de tuyère et les trois axes du RCS.

| Paramètre | Modèle d’imitation de phase 4 |
|---|---|
| Données d’imitation | 151 vols réussis sur 160, soit 45 662 exemples |
| Corrections DAgger retenues | 2 tours de 32 départs, soit 67 704 exemples après agrégation |
| Entraînement | 35 époques d’imitation, puis 15 époques par tour DAgger |
| Optimiseur | AdamW, lots de 256, taux de 0,001 en imitation puis 0,0006 en DAgger |
| Sélection | Graine d’entraînement 1, checkpoint du tour 2, retenu sur 40 départs de comparaison |

Sur le jeu de validation de 20 départs, cette graine passe de **11/20** après imitation à **17/20**, puis **19/20** après les deux premiers tours DAgger. Un troisième tour a également été suivi dans les expériences. Le checkpoint officiel vient du deuxième tour, choisi sur les 40 départs de comparaison. [Collecte des données](src/rocket_quentin/dagger/phase_demonstrations.py) · [Résultats par graine](README.md)

## Comment on vérifie qu’il pilote seul

Dans une comparaison sur 40 nouveaux départs, le réseau réussit **36 poses avec les observations courantes**. Il obtient **0/40** en recevant l’observation initiale à chaque pas. Quand on reprend une observation réelle à 1,90 m du pont et qu’on change uniquement la vitesse de descente indiquée au réseau, il commande **13,75 kN** à 3,06 m/s vers le bas et coupe le moteur à 1,56 m/s. Le réseau répond donc bien à l’évolution de l’état du vol.

Le pilote de référence réussit **40/40** sur ces mêmes départs. L’expérience établit que le modèle appris réagit aux observations. Démontrer que le réseau découvre une stratégie plus efficace par récompense demanderait une comparaison dédiée en apprentissage par renforcement.

## Résultats officiels et robustesse

| Phase | Situation | Modèle évalué | Poses sur 100 | Seuil |
|---|---|---|---:|---:|
| 0 | Barge fixe | DAgger de phase 0 | 96 | 95 |
| 1 | Barge mobile | Modèle DAgger de phase 0 transféré | 93 | 90 |
| 2 | Carburant et poussée minimale | Réseau de phase 2 entraîné par imitation et DAgger | 96 | 85 |
| 3 | Vagues | Réseau de phase 3 entraîné par imitation et DAgger | 88 | 80 |
| 4 | Vent et RCS | Réseau de phase 4 entraîné par imitation et DAgger | 89 | 75 |

Les graines de collecte et de correction servent à apprendre, celles de validation et de comparaison servent à choisir les poids, et les graines officielles **10 000 à 10 099** servent au test final. Sur les 100 vols officiels de phase 4, les 11 échecs comprennent 10 sorties dues à l’inclinaison et une chute à l’eau. [Contrat commun](../../CONTRACT.md)

Sur un jeu supplémentaire de 40 départs communs aux trois conditions, le même modèle donne **38/40** en conditions normales, **39/40** avec des rafales d’amplitude doublée et **38/40** avec un bruit ajouté à des mesures de position, vitesse et vent. Le résultat soutient une résistance à ces deux perturbations précises, appliquées séparément. La différence d’une pose sur 40 reste faible. Dans nos expériences, SAC favorise l’exploration pendant l’apprentissage et DAgger entraîne le réseau sur ses propres écarts. Les tests avec des perturbations mesurent la robustesse du modèle retenu.

Pour le rapport du cours, SAC relève de nos **expériences d’apprentissage par renforcement**, et DAgger de **l’imitation supervisée**. Le sujet demande de comparer plusieurs algorithmes de RL. Le dossier de Quentin documente SAC comme algorithme de RL testé, et le bilan de cette exigence se fait à partir des expériences de toute l’équipe. [Sujet du cours](../../docs/index.html)

## Rejouer le test ou retrouver les essais

Depuis la racine du dépôt, l’évaluation officielle du modèle de phase 4 se lance avec cette commande.

```powershell
uv run --locked python -m rocket_quentin.evaluate --phase 4 --model players/quentin/models/dagger/phase4-autonomous.pt
```

Pour revoir les 100 vols du candidat SAC de phase 0 tant que le checkpoint historique est disponible localement.

```powershell
uv run --locked python -m rocket_quentin.evaluate --run-dir "players/quentin/runs/imitation-s0-xk5eiozj/checkpoints/sac-04000"
```

Le [README](README.md) contient les commandes de collecte, d’entraînement et d’évaluation par phase. Les résultats détaillés et les journaux W&B hors ligne sont dans `players/quentin/runs/`. Les modèles retenus se trouvent dans `players/quentin/models/`.
