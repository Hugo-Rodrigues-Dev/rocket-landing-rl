# Contrat commun

Ce contrat fixe **ce qui doit être identique** pour que nos trois approches soient comparables : le scénario physique, les phases, les critères de réussite et les mesures.

Tout le reste est libre : représentation de l'état, récompense, algorithme, réseau, curriculum, pas de simulation, rendu…

Le contrat ne change que par une PR approuvée par les deux autres.

---

## 1. Le problème

Une fusée descend vers une barge et doit s'y poser en sécurité. Un seul agent, un seul objectif.

Référence détaillée de la phase 1 : [`docs/mdp-etape-1.pdf`](docs/mdp-etape-1.pdf).

### Ce qui est fixé

Unités SI, repère du monde avec `z` vers le haut, `g = 9,81 m/s²`.

| Élément | Valeur |
|---|---|
| Fusée | cylindre de 10 m × 1 m, masse à vide 1 000 kg, 4 jambes d'atterrissage |
| Moteur | un seul, au bas de la fusée, poussée de 0 à `T_max = 19 620 N` (2 × poids à vide) |
| Tuyère | orientable de ± 10° sur deux axes, dans le repère de la fusée |
| Barge | pont de 15 × 15 m, à 1 m au-dessus de la mer |
| Fréquence de décision | **20 Hz au maximum** (une action toutes les 0,05 s ou plus) |
| Durée max d'un épisode | 30 s de temps simulé |

### Ce que l'agent peut observer

Toute grandeur qu'un vrai système embarqué pourrait mesurer : position, vitesse, orientation et vitesse angulaire de la fusée, position et vitesse de la barge, contacts des jambes, carburant restant. La représentation est libre (quaternion, matrice de rotation, repère relatif…).

**Interdit** : observer l'avenir, comme la trajectoire future de la barge ou les rafales à venir.

### Ce que l'agent peut commander

Seulement les actionneurs de la phase en cours : poussée et tuyère, puis RCS à partir de la phase 4. Le format est libre : actions continues, ou discrétisées si on préfère.

### Départ (tiré au hasard à chaque épisode)

| Grandeur | Plage |
|---|---|
| Altitude au-dessus du pont | 80 à 120 m |
| Décalage horizontal par rapport au centre du pont | disque de rayon 30 m |
| Vitesse verticale | −15 à −5 m/s |
| Vitesse horizontale | disque de rayon 3 m/s |
| Inclinaison | 0 à 10°, direction au hasard |
| Roulis (rotation autour de l'axe de la fusée) | 0 à 360° |

### Fin d'épisode

| Issue | Condition |
|---|---|
| **Posée** (succès) | 4 jambes sur le pont, vitesse relative au pont < 0,2 m/s, vitesse angulaire < 0,1 rad/s |
| Atterrissage dur | une jambe touche le pont avec une vitesse verticale relative > 2 m/s |
| Crash | le corps de la fusée (pas une jambe) touche le pont ou la mer |
| À l'eau | la base de la fusée descend sous le niveau du pont, en dehors du pont |
| Hors zone | à plus de 100 m de la barge, plus haut que 200 m, ou inclinaison > 60° |
| Temps écoulé | 30 s (troncature, pas un échec de physique) |

---

## 2. Les phases

Chaque phase **ajoute** quelque chose à la précédente.

| Phase | Ajout | Seuil de validation |
|---|---|---|
| **0** | Barge **immobile** | ≥ 95 % posées |
| **1** | Barge mobile : vitesse horizontale aléatoire et lissée (Ornstein-Uhlenbeck, θ = 0,5 s⁻¹, σ = 1 m·s⁻¹ᐟ², bornée à 3 m/s) | ≥ 90 % |
| **2** | **Carburant** : 80 kg au départ, consommation `ṁ = T / (300 s × g)`, la masse diminue. **Poussée minimale 60 %** quand le moteur tourne (moteur éteint en dessous), donc le vol stationnaire devient impossible. Réservoir vide = moteur éteint | ≥ 85 % |
| **3** | **Vagues** : le pont monte et descend (± 0,5 m, période 8 s) et bascule (± 3° en tangage et en roulis, périodes de 6 à 10 s, phases au hasard) | ≥ 80 % |
| **4** | **Vent** : vent moyen de 0 à 8 m/s dans une direction au hasard, rafales de ± 3 m/s, traînée `½ ρ C_d A v²` avec ρ = 1,225, C_d = 1, A = 10 m². **RCS** : couples de ± 1 000 N·m en tangage et en lacet, ± 100 N·m en roulis | ≥ 75 % |
| **Bonus** | Bruit sur les capteurs, départ de 200 à 300 m, ou toute autre difficulté proposée par PR | — |

Les seuils et les valeurs des phases 2 à 4 sont des propositions de départ. On les ajustera par PR si une phase se révèle irréaliste.

---

## 3. Évaluer un agent

- **100 épisodes** avec les graines **10 000 à 10 099**. Ces graines sont **réservées à l'évaluation**, jamais utilisées pour entraîner.
- La politique est jouée en **mode déterministe** : l'action moyenne, sans tirage aléatoire.
- Une phase est **validée** quand le taux de réussite atteint son seuil.

### Mesures à rapporter

| Mesure | Définition |
|---|---|
| Taux de réussite | % d'épisodes terminés en « posée » |
| Précision | distance horizontale **médiane** entre la fusée et le centre du pont, sur les épisodes réussis |
| Vitesse au contact | vitesse verticale relative **moyenne** au premier contact d'une jambe |
| Inclinaison au contact | inclinaison **moyenne** au premier contact |
| Causes d'échec | nombre d'épisodes par issue (atterrissage dur, crash, à l'eau, hors zone, temps écoulé) |
| Carburant restant | moyenne sur les épisodes réussis (phase 2 et au-delà) |

### Classement

1. la phase validée la plus haute ;
2. à égalité : le taux de réussite sur cette phase ;
3. puis la précision.

---

## 4. Ce que chacun livre, par phase validée

Dans `players/<prenom>/` :

- une **commande pour reproduire l'évaluation** (documentée dans son `README.md`) ;
- le **modèle entraîné** dans `models/`, s'il pèse moins de 20 Mo ;
- un **tableau des mesures** ci-dessus dans son `README.md`, avec la configuration d'entraînement (algorithme, réglages, nombre de pas, graines) ;
- les **runs dans le projet WandB commun** (voir ci-dessous).

## 5. WandB

- Un projet commun `rocket-landing-rl`, dans l'équipe WandB du groupe.
- Pour chaque run : `group` = son prénom, `tags` = phase (`phase0`…) et algorithme (`sac`, `ppo`…).
- Tous les runs comptent, y compris les échecs : le sujet note le nombre et la méthode des expériences.
