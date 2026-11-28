# Rocket Landing RL — guide de Hugo

## 1. Où on en est

La branche `hugo/env-phase0` contient un modèle vertical analytique et une scène MuJoCo 3D propre
à Hugo. Les commandes de lancement des deux démos sont dans le [README](README.md). Le modèle
analytique s'arrête au premier contact. Dans la scène 3D, après un impact doux, on peut poursuivre
la simulation pour que les jambes prennent appui sur le pont.

L'environnement Gymnasium tire des départs au-dessus de la barge immobile selon le
[contrat commun](../../CONTRACT.md). Un premier réseau apprend la poussée sur l'exercice vertical.
L'entraînement et l'évaluation de la phase 0 complète restent à mettre en place.
Le [MDP de référence](../../docs/mdp-etape-1.pdf) décrit surtout la barge mobile de la phase 1.
Utilise le contrat actuel pour les règles et les seuils d'évaluation.

### À lancer maintenant

Dans PowerShell, depuis la racine de `rocket-landing-rl` :

```powershell
Push-Location players/hugo/src
try {
    python -m rocket_hugo.demo_vertical
    python -m unittest discover -s ../tests -p test_vertical.py -v
} finally {
    Pop-Location
}
```

Ces deux commandes utilisent la bibliothèque standard de Python. La démo affiche cette comparaison
pour une fusée qui part à **100 m**, en descendant à **10 m/s** :

| Commande | Temps avant contact | Vitesse verticale juste avant le contact |
|---|---:|---:|
| Moteur coupé | 3,61 s | −45,41 m/s |
| Poussée égale au poids | 10,00 s | −10,00 m/s |
| Freinage selon la hauteur et la vitesse | 16,26 s | −0,12 m/s |

Pour voir un rendu 3D dès maintenant, regarde le [rejeu du pilote de Mathis](https://github.com/Hugo-Rodrigues-Dev/rocket-landing-rl/blob/mathis/env-phase0/players/mathis/docs/captures/etape-5/pd-reference.gif)
et [son rejeu d'un PPO qui plane](https://github.com/Hugo-Rodrigues-Dev/rocket-landing-rl/blob/mathis/env-phase0/players/mathis/docs/captures/etape-5/v1-ppo.gif).
Dans le premier, un pilote écrit à la main décide des commandes. Dans le second, les commandes
viennent d'un réseau entraîné. Observe ce que chacun fait à l'approche du pont.

## 2. Lire les fichiers dans l'ordre

| Fichier ou dossier | Rôle actuel |
|---|---|
| [src/rocket_hugo/vertical.py](src/rocket_hugo/vertical.py) | Décrit l'état et calcule le mouvement pour une commande. |
| [src/rocket_hugo/demo_vertical.py](src/rocket_hugo/demo_vertical.py) | Choisit les commandes, fait tourner le vol et affiche le résultat. |
| [tests/test_vertical.py](tests/test_vertical.py) | Construit trois situations précises pour vérifier la physique. |
| [tests/test_setup.py](tests/test_setup.py) | Vérifie les imports de la pile et fait chuter un objet dans un petit modèle MuJoCo sans fusée ni barge. |
| [src/rocket_hugo/__init__.py](src/rocket_hugo/__init__.py) | Point d'entrée du paquet Python, actuellement une description. |
| [pyproject.toml](pyproject.toml) | Déclare le paquet `rocket_hugo`, Python 3.12 et ses dépendances. Le dépôt partage l'environnement Python installé par `uv`. |
| [README.md](README.md) | Point de départ pour lancer le code et retrouver les résultats. |
| `models/` | Emplacement des futurs modèles finaux. |

### `vertical.py` — calculer ce que fait la fusée

1. **Les constantes** donnent la masse (1 000 kg), la gravité (9,81 m/s²), la poussée maximale
   (19 620 N) et le temps entre deux décisions (0,05 s). 600 décisions complètes correspondent à
   30 s. La démo utilise cette limite pour arrêter sa boucle.
2. **`VerticalState`** contient deux nombres. `height_m` est la hauteur au-dessus du pont,
   dont le niveau vaut zéro. `velocity_m_s` est positive vers le haut et négative en descente.
   La classe est immuable. Chaque appel à `advance` construit l'état suivant.
3. **`thrust_newtons(action)`** transforme une commande sans unité en poussée physique. Pour
   `−1`, elle vaut 0 N. Pour `0`, elle vaut 9 810 N. Pour `+1`, elle vaut 19 620 N.
   La fonction lève une erreur si la commande est hors de l'intervalle ou non finie.
4. **`advance(state, action)`** calcule l'accélération `a = poussée / masse − gravité`.
   Pendant un pas, la poussée reste constante. En l'absence de contact, après un temps `dt`,
   la vitesse devient `v + a × dt` et la hauteur devient `h + v × dt + a × dt² / 2`.
5. **Le contact** peut arriver avant la fin du pas. Le code résout l'équation « hauteur = 0 »
   à partir de l'état de départ et garde le premier contact s'il arrive dans les 0,05 s.
   Il renvoie alors la hauteur zéro, la vitesse juste avant le choc et le temps exact écoulé.
   Cette expérience prend fin à l'impact. Les rebonds, les jambes et le frottement relèveront
   de la scène 3D.

**Essaie un pas à la main.** Départ à 100 m, vitesse −10 m/s, action `0`. La poussée égale
le poids, donc l'accélération vaut zéro. La fusée descend de 0,5 m en 0,05 s. Le nouvel état
vaut `(99.5, -10.0)`. Avec l'action `+1` au même départ, elle descend encore pendant ce pas
mais termine à environ −9,51 m/s. Une accélération vers le haut freine d'abord la vitesse vers
le bas.

### `demo_vertical.py` — choisir la commande et répéter

`main()` rejoue le même état initial trois fois. Dans les deux premiers vols, la commande
reste fixe. Pour le troisième, la boucle appelle `braking_action(state)` à chaque décision,
passe le résultat à `advance`, remplace l'état, ajoute le temps écoulé et recommence jusqu'au
contact ou à la limite. Le champ `None` associé au scénario « freinage » signifie dans cette
boucle que la commande doit être recalculée.

`braking_action` choisit d'abord une vitesse de descente visée. À 100 m, elle est de
−12 m/s. À 10 m, elle est de −5 m/s. Tout près du pont, elle vise −0,12 m/s pour finir
la descente. Le code soustrait la vitesse actuelle à cette consigne et divise l'écart par
0,25 s. Si la fusée descend à −8 m/s alors qu'elle vise −5 m/s, le calcul demande une
accélération vers le haut de +12 m/s². La conversion en commande et la saturation à `+1`
limitent l'accélération aux capacités du modèle. Cette règle fournit un exemple de **politique**,
une façon de choisir l'action à partir de ce que l'on observe.

### `test_vertical.py` — vérifier trois choses différentes

| Test | Ce qu'il permet de détecter |
|---|---|
| Poussée égale au poids | Une erreur d'unité, de signe ou de pas de temps. À −10 m/s, la vitesse doit rester à −10 m/s et la hauteur passer de 100 à 99,5 m. |
| Chute libre à 1 cm | Le premier impact doit arriver à environ 0,045 s, avant la prochaine décision. Le test compare sa durée et sa vitesse à la solution de chute libre. |
| Poussée maximale tout près du pont | La trajectoire mathématique finirait le pas au-dessus du pont après avoir traversé son niveau. Le test impose de détecter le premier passage à hauteur zéro. |

## 3. De la physique à l'apprentissage

### À quoi servent les bibliothèques ?

| Outil | Ce qu'il fait dans notre projet |
|---|---|
| Python | Exécute notre code et relie les différents éléments. |
| MuJoCo | Calcule comment les objets bougent et entrent en contact sous l'effet des forces. Il peut aussi afficher la scène en 3D. |
| Gymnasium | Définit le dialogue entre la fusée simulée et l'agent. On initialise un épisode, puis on fournit une action et on récupère une observation, des points et les informations de fin. |
| PyTorch | Fournit les réseaux et le calcul des gradients qui permettent de modifier leurs paramètres. |
| Stable-Baselines3 (SB3) | Fournit les algorithmes qui collectent des expériences et utilisent PyTorch pour entraîner les réseaux. |
| Weights & Biases (WandB) | Enregistre les réglages et les courbes des essais pour retrouver et comparer les résultats. |

**MuJoCo.** On décrira la mer, le pont, le corps de la fusée et ses jambes dans un fichier
`scene.xml` au format MJCF. Python chargera cette description dans un `MjModel`. Pendant
l'expérience, `MjData` contiendra les positions, les vitesses et les contacts qui évoluent.
On applique la poussée, puis `mj_step` fait avancer la physique d'un sous-pas. Par exemple,
avec un sous-pas de 5 ms, on conserve la même commande pendant dix sous-pas pour obtenir une
décision toutes les 50 ms. Cette valeur de 5 ms est un point de départ à vérifier.
[Documentation de MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html)

**Gymnasium.** Notre futur `env.py` déclarera la forme des observations et des actions.
`reset(seed=...)` initialise un vol et renvoie `(observation, info)`. Une graine fixe permet
de reproduire les tirages d'une même configuration. `step(action)` applique la commande à
MuJoCo, relit les mesures et renvoie
`(observation, reward, terminated, truncated, info)`. Par exemple, une posée ou un crash
termine l'épisode (`terminated`). Le délai de 30 s tronque un épisode encore en cours
(`truncated`). `info` peut exposer le motif et la vitesse au premier contact pour l'analyse.
L'agent choisira ses commandes à partir du vecteur d'observation.
[API Gymnasium](https://gymnasium.farama.org/api/env/)

### Le MDP, en suivant une décision

| Brique | Exemple concret |
|---|---|
| État initial | On tire une hauteur, un décalage, des vitesses et une orientation au début du vol. |
| État physique et observation | Le simulateur garde son état complet. L'agent reçoit les mesures utiles, par exemple la vitesse relative au pont, l'orientation et les contacts. |
| Action | Trois valeurs continues entre −1 et +1 règlent la poussée et les deux axes de tuyère. |
| Transition | La scène avance sous l'effet de cette action, de la gravité et des contacts. En phase 1 s'ajoute le tirage qui fait évoluer la barge. |
| Récompense | On attribue un score pour cette transition. Une version simple peut donner un bonus lors d'une posée, une pénalité pour un crash et zéro pendant le vol. |
| Actualisation `gamma` | Une récompense reçue dans `k` décisions pèse `gamma^k` dans le retour. Avec 0,995 et 20 décisions/s, +100 reçu après 1 s vaut environ +90,5 vu du départ. |

Un **épisode** est un vol. Une **transition** est une décision et son résultat. Le **retour**
est la somme des récompenses futures pondérées. Le **taux de réussite** compte les posées selon
le contrat. Une courbe de retour peut monter parce que l'agent évite les pénalités en planant.
Il faut regarder aussi les posées, les temps écoulés et les vidéos.

### Comment les commandes du réseau s'améliorent

1. **Épisode initial.** Depuis un départ tiré, le réseau (l'**acteur**) lit les mesures et
   propose des commandes. Pendant l'entraînement, on essaie différentes actions pour explorer.
   Les premiers vols peuvent échouer très vite.
2. **Collecte.** Gymnasium fournit une nouvelle observation et une récompense après chaque
   action. L'algorithme garde les transitions et les fins de vol.
3. **Estimation.** Un **critique** estime ce que ces décisions peuvent rapporter à terme.
   Une action de freinage peut être intéressante parce qu'elle prépare une pose quelques
   secondes plus tard, même avec zéro point immédiat.
4. **Mise à jour.** L'optimiseur ajuste les poids des réseaux à partir de ces estimations.
   **PPO** se sert de vols collectés avec la politique récente et limite l'effet de changements
   trop grands de probabilité des actions. **SAC** réutilise des transitions d'anciens vols
   stockées dans un tampon et entraîne l'acteur à chercher un bon retour tout en favorisant
   l'exploration. [PPO dans SB3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) ·
   [SAC dans SB3](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
5. **Comparaison.** On gèle des versions du réseau à différents nombres de décisions. On les
   rejoue en mode déterministe sur les mêmes départs de développement et on compare les issues.
   Une fois la configuration choisie, on mesure le résultat sur les graines officielles.

Exemple de diagnostic. Au début, le réseau coupe la poussée et touche trop vite. Le modèle
suivant freine mais flotte jusqu'à 30 s. Le troisième touche doucement mais hors du centre.
Avec la même caméra et le même départ, ces trois vidéos permettent de relier le type d'échec
à un tracé, à des commandes et à la récompense reçue.

### Ce que fait maintenant `env.py`

1. `reset()` prépare la fusée. Le mode `contract` échantillonne les grandeurs de départ, alors
   que `vertical` place la fusée droite à 100 m au-dessus du centre à −10 m/s. La démo isole
   ainsi l'effet de la commande. Une même graine permet de comparer des commandes à départ identique.
2. `step()` convertit la première valeur en newtons. Les deux autres font tourner le vecteur
   de poussée dans le repère de la fusée. Le code exprime ensuite cette force dans le repère du
   monde. Appliquée à la tuyère, elle agit à distance du centre de masse et crée un couple.
   Avec `--mode gimbal`, on voit la fusée se déporter et tourner.
3. Dix sous-pas de 5 ms font une décision de 50 ms, sauf fin en cours de décision. Le code
   lit les contacts avant la résolution du choc pour conserver la vitesse au premier contact.
4. L'observation contient **19 valeurs** : position relative divisée par 100 m, vitesse du point
   de base divisée par 20 m/s, deux axes de la fusée (six valeurs), vitesse angulaire divisée par
   2 rad/s et quatre contacts. L'agent travaille avec ces mesures.
5. La récompense de référence donne **+100** pour une pose, **−100** pour un échec physique,
   **−25** pour la limite de temps et **0** pendant le vol. Comme expirer coûte moins cher qu'un
   crash, un futur agent pourrait préférer rester en l'air. On suivra les issues pour le détecter.

La règle de démonstration pilote la descente droite. L'environnement laisse la tuyère et la
poussée disponibles à chaque décision, y compris après un contact doux tant que le vol continue.

### Utiliser la fenêtre sans toucher à la physique

La démo Python choisit les commandes et fait avancer MuJoCo. La fenêtre affiche le résultat.
Les scénarios numérotés du [README](README.md) sont des expériences avec des commandes fixes ou
des règles écrites à la main. Le cas 2 démontre qu'une accélération nulle conserve une vitesse
existante. Les cas 4 et 5 isolent l'effet d'une correction d'angle : mêmes 5° au départ,
même règle de freinage, seule l'orientation de la tuyère change selon l'angle et la vitesse
de rotation. Le redressement vise la verticale sur un seul axe. Le guidage vers le centre et
l'apprentissage d'une politique en 3D restent des expériences distinctes.

Appuie sur `P`, puis sur `N` plusieurs fois pour lire les effets d'une décision. L'affichage
donne le pourcentage de poussée appliquée au pas précédent. L'inclinaison décrit la fusée.
Les deux angles de tuyère décrivent la commande du moteur. La fusée ne change pas d'orientation
instantanément quand la tuyère change de direction : elle reçoit un couple, commence à tourner,
puis le pilote ajuste la commande.

Dans le visualiseur natif, `Watch` sert à inspecter des valeurs internes comme `qpos` et
`qvel`. `Physics` touche à la gravité, aux contacts ou au solveur. Pour garder des essais
comparables, les paramètres physiques viennent de `scene.xml`. `Rendering`, `Visualization`
et les groupes de géométrie règlent ce qui est affiché. La démo fournit déjà les mesures utiles
dans la fenêtre. Les courbes d'entraînement viendront des transitions Python et de WandB.
[Code du visualiseur MuJoCo](https://github.com/google-deepmind/mujoco/blob/4.0.1/simulate/simulate.cc)

### Combien d'expériences faut-il ?

Le simulateur **génère les données pendant l'entraînement**. Une ligne d'expérience correspond
à une décision : observation avant, commande, récompense, observation après et indicateurs de
fin. SAC réutilise les transitions stockées lors de plusieurs mises à jour du réseau.

Dans notre premier essai PPO, le réseau reçoit la hauteur et la vitesse, puis produit une
commande de poussée. Pendant la collecte, la commande est échantillonnée pour explorer.
Après 512 décisions, PPO s'appuie sur la récompense et la valeur estimée par le critique
pour mettre le réseau à jour. Au rejeu, on prend la commande déterministe du réseau sauvegardé.
L'environnement attribue un score continu au contact : un impact plus lent rapporte plus,
même si les deux impacts sont au-dessus du seuil de réussite. Il ajoute un petit coût par
décision, plus fort quand la vitesse s'éloigne de la consigne de descente. Cette consigne sert
à noter le vol. PPO choisit lui-même la poussée à partir de ses observations.

Les [résultats du premier essai](README.md#rejouer-le-premier-apprentissage-ppo) montrent pourquoi
il faut garder plusieurs versions. Les trois graines passent le seuil à 24 576 décisions sur
les départs de développement. L'une repasse juste au-dessus à 32 768 décisions. Pour regarder
ce recul, rejoue les versions 4 et 5 de `vertical-ppo-s1` puis compare la vitesse au contact.

À 20 décisions par seconde simulée, **100 000 décisions représentent environ 5 000 secondes
de vol simulé**. Comme un épisode dure au plus 600 décisions, ce budget couvre au moins
167 vols commencés, souvent davantage quand ils échouent tôt. Le temps réel dépend du débit
de la simulation, des mises à jour et des évaluations. À **200 décisions/s** mesurées sur
l'entraînement complet, 100 000 décisions prendraient **8 min 20 s**, et un million **1 h 23 min**.
Ce débit est un exemple de calcul. Il faut mesurer celui de la machine qui entraîne le modèle.

Commencer par quelques milliers de décisions pour vérifier la collecte et le rejeu. Ensuite,
sauvegarder des checkpoints et comparer sur les mêmes départs de développement. Si les poses
restent à zéro alors que tous les impacts sont durs, regarder d'abord l'exploration, la
récompense et les départs d'entraînement avant d'augmenter le budget.

## 4. Architecture actuelle et fichiers à ajouter

| Emplacement prévu | Responsabilité |
|---|---|
| `src/rocket_hugo/assets/scene.xml` (créé) | Géométrie, masses, contacts et configuration physique de base. |
| `src/rocket_hugo/env.py` (créé) | `reset`, `step`, observations, décisions, issues et mesures physiques. |
| `src/rocket_hugo/demo_3d.py` (créé) | Rejeu des commandes de diagnostic avec ou sans fenêtre MuJoCo. |
| `src/rocket_hugo/vertical_env.py` (créé) | Enveloppe Gymnasium de la référence analytique, avec deux observations, une action et le score de l'exercice. |
| `src/rocket_hugo/train_vertical.py` (créé) | PPO sur CPU. Enregistre le réseau initial et les versions successives. Mesure chaque version sur les mêmes départs. |
| `src/rocket_hugo/replay_vertical.py` (créé) | Charge les versions évaluées et dessine le même vol vertical dans MuJoCo, pas à pas ou en lecture continue. |
| `src/rocket_hugo/baseline.py` | Pilote manuel pour diagnostiquer la scène et établir un point de comparaison. |
| `src/rocket_hugo/train.py` | Choisit la configuration, lance PPO ou SAC, sauvegarde modèle et normalisation, enregistre les mesures. |
| `src/rocket_hugo/evaluate.py` | Rejoue les politiques avec des graines explicites et calcule les métriques du contrat. |
| `src/rocket_hugo/replay.py` | Rejoue un départ et une politique avec la caméra, les commandes et les mesures. |
| `tests/`, `models/`, `runs/` | Vérifications, éventuels modèles finaux et résultats locaux des essais. |

Ces noms sont des repères. Créer chaque fichier quand son premier usage arrive. Extraire
la récompense ou la configuration dans un autre module si elles deviennent difficiles à lire
dans `env.py`. Garder les scènes de diagnostic et les départs du contrat accessibles par des
options distinctes.

## 5. Checklist de réalisation

Les commandes du §1 fonctionnent avec le code actuel. Les commandes `train`, `evaluate` et
`replay` ci-dessous désignent des interfaces à construire. Pour chaque étape, garder au moins
un exemple reproductible avec son départ, ses commandes et son résultat attendu.

### Construire et vérifier la simulation

- [x] **0. Référence verticale.** Rejouer les trois commandes. Vérifier la première décision à
  la main, puis lancer les trois tests. À observer dans la démo actuelle → la vitesse change avec
  la poussée et le freinage réduit la vitesse au contact.
- [ ] **1. Installer la pile.** Depuis la racine, lancer `uv sync --all-packages --locked`, puis
  `uv run --locked pytest players/hugo/tests/test_setup.py`. On doit pouvoir importer la pile
  et obtenir la chute libre attendue dans MuJoCo. En cas de souci d'installation, la référence
  verticale du §1 reste exécutable. Pour voir une scène MuJoCo indépendamment du dépôt, ouvrir le
  [visualiseur officiel](https://live.mujoco.org/).
- [x] **2. Faire la scène 3D immobile.** Construire le pont, la mer, la fusée libre et ses quatre
  pieds. Vérifier les dimensions, la masse et quels objets ont le droit de toucher le pont. Rendu
  → ouvrir la future scène avec `python -m mujoco.viewer --mjcf <chemin-vers-scene.xml>` dans
  l'environnement installé. Faire tomber la fusée à quelques centimètres du pont et regarder quels
  pieds touchent en premier.
- [x] **3. Brancher le moteur.** Une commande impose une intensité et deux angles. Faire d'abord
  les trois essais verticaux connus et comparer les trajectoires avec `vertical.py` avant contact,
  avec une tolérance liée au sous-pas. Ensuite, orienter la tuyère sur un axe à la fois et vérifier
  les signes du déplacement et de la rotation. Rendu → comparer `--mode weight` et `--mode gimbal`.
  Il restera utile d'ajouter des tracés pour analyser les politiques apprises.
- [ ] **4. Terminer la vérification de l'environnement Gymnasium de phase 0.** Le vérificateur
  Gymnasium, les tests de décision et le contact vertical sont en place. Il reste à couvrir les
  cas limites, notamment bord du pont, arrivée inclinée et plusieurs issues au même sous-pas.
  Après installation de la pile complète, lancer aussi le vérificateur SB3. Rendu → comparer
  une chute centrée, un atterrissage dur, un crash et un départ à côté du pont.
- [ ] **5. Créer deux références et l'évaluation.** Comparer actions aléatoires et pilote
  programmé. Construire le pilote par étapes → freinage vertical, redressement, guidage latéral.
  Un pilote capable de réussir sur des départs variés aide à diagnostiquer un mauvais résultat de
  PPO ou SAC. Rendu → sur le même départ, superposer le point de départ, la trajectoire, la trace
  de vitesse et le premier contact. Faire fonctionner `evaluate.py` avec les deux politiques.

### Apprendre et mesurer

- [x] **6. Entraîner un premier agent sur une variante verticale Gymnasium.** Réutiliser la
  référence verticale pour créer un exercice reproductible avec deux observations et une action.
  Définir un succès adapté à cet exercice. Faire un essai court avec SAC ou PPO et vérifier qu'il
  collecte des épisodes, sauvegarde un checkpoint et recharge la même politique. Rendu → rejouer
  le réseau initial et des checkpoints sur les mêmes graines de développement. Suivre la vitesse
  au contact et les réussites. Ce score sert à comprendre le mécanisme.
- [ ] **7. Passer aux trois actions dans MuJoCo.** Commencer par des départs faciles dans la
  scène complète. Comparer une récompense de base à une variante qui encourage une amélioration
  mesurable de l'approche. Garder ces deux variantes distinctes et journaliser leurs composantes.
  Rendu → examiner autant les vols qui échouent que les poses. Si le retour monte alors que les
  temps écoulés augmentent, revoir le signal de récompense et les trajectoires.
- [ ] **8. Augmenter la difficulté et mesurer le transfert.** Introduire l'altitude, puis le
  décalage latéral et l'inclinaison dans des expériences où une seule difficulté évolue à la fois.
  Ensuite, tester les trois ensemble jusqu'aux départs complets. Conserver des scénarios fixes
  faciles et difficiles pour mesurer les progrès et les régressions. Rendu → comparer les
  checkpoints sur ces mêmes scénarios, même quand le curriculum change.
- [ ] **9. Comparer les expériences proprement.** Prendre comme premier protocole trois graines
  d'entraînement par configuration, un budget comparable en décisions et les mêmes épisodes de
  développement. Comparer SAC et PPO, puis deux architectures PyTorch ou réglages justifiés.
  Garder un changement à la fois pour les comparaisons qui doivent expliquer une différence.
  Journaliser dans WandB les taux d'issue, la difficulté et les mesures physiques en plus du
  retour. Afficher chaque graine et la dispersion. Séparer le temps d'entraînement du nombre
  de décisions simulées.
- [ ] **10. Valider la phase 0.** Geler le modèle et les éventuelles statistiques de normalisation.
  L'évaluer en mode déterministe sur les **100 graines 10 000 à 10 099**. Garder ces graines hors de
  l'entraînement, des promotions du curriculum et du choix des checkpoints. Le contrat demande
  **95 posées sur 100**. Rapporter la précision médiane sur les réussites, la vitesse et
  l'inclinaison moyennes au premier contact, et les causes d'échec. Rendu → rejouer une réussite et
  quelques causes fréquentes d'échec depuis le rapport.

### Étendre les phases et livrer

| Étape | À construire et à vérifier | Ce qu'on veut voir |
|---|---|---|
| ☐ Phase 1 · **90 %** | Barge mobile. Vérifier vitesse visée bornée à 3 m/s, accélération limitée à 0,5 m/s² et entraînement d'une fusée posée par frottement. | Trace de la barge et de la fusée, vitesses relatives et pose pendant la dérive. |
| ☐ Phase 2 · **85 %** | Ajouter les 80 kg de carburant, la masse variable, la consommation et la poussée minimale de 60 % lorsque le moteur fonctionne. Tester la coupure à réservoir vide. | Courbes de poussée et de carburant, avec jauge dans le rejeu. |
| ☐ Phase 3 · **80 %** | Ajouter la hauteur et l'inclinaison du pont selon le contrat. Vérifier les amplitudes, les périodes et les vitesses relatives au pont. | Vue latérale du pont oscillant et pose rejouée avec les angles du pont. |
| ☐ Phase 4 · **75 %** | Ajouter vent, rafales, traînée et commandes RCS. Vérifier le sens des forces et les limites des couples. | Trajectoires avec et sans vent sur le même départ et affichage des commandes RCS. |

- [ ] **11. Répéter l'évaluation par phase.** Appliquer le protocole du contrat à chaque phase
  et conserver les résultats des expériences. À partir de la phase 2, inclure la quantité moyenne
  de carburant restant sur les épisodes réussis.
- [ ] **12. Rendre les résultats reproductibles.** Ajouter dans le README les commandes exactes,
  le tableau des mesures, les réglages, les graines et les liens WandB. Le projet commun s'appelle
  `rocket-landing-rl`, avec le groupe `hugo` et les tags de phase et d'algorithme. Garder tous les
  essais. Publier dans `models/` seulement les modèles finaux des phases validées de moins de
  20 Mo, suivant les règles du dépôt.
- [ ] **13. Préparer le rendu et la PR.** Vérifier `uv run --locked ruff check .` et
  `uv run --locked pytest players/hugo`. Dans le rapport et la présentation, montrer comment
  fonctionne l'environnement, justifier les réseaux et les algorithmes, comparer les expériences
  et expliquer les échecs avec des courbes et des rejeux. La PR part d'une branche `hugo/…` vers
  `main`, avec les changements de `players/hugo`.

À la fin de chaque étape, noter quatre choses dans le README ou le journal d'expérience.
**Ce qui a changé, ce que le code calcule, comment le reproduire et ce que le résultat permet de
conclure.** Par exemple, le contact à −0,12 m/s confirme que la règle sait freiner sur ce départ
vertical. L'évaluation des quatre jambes sur les 100 départs du contrat viendra avec la phase 0.
