# simulation-onde
Je présente ici un code Python permettant de créer une animation d'onde.

Nous pouvons choisir à notre guise les conditions aux bords (périodiques, d'absorption ou réflectives) et le système d'étude (fente unique, double fentes, lentille convergente, accélération de la source ou pluie) en enlevant les "#" des appels de fonctions depuis la fonction update et depuis les fonctions de célérité.

Une fois le code lancé et les calculs terminés, le code renvoie chaque image de l'animation (30FPS, que l'on peut modifier) d'une durée de 20 secondes (que l'on peut modifier via la variable tf) dans le dossier "img" (à créer prélablement).

Afin de créer une animation à partir de ces images, vous devez disposer du logiciel FFMPEG sur votre ordinateur et exécuter la commande mentionnée en bas du code (en tant que commentaire). Pour ce faire, ouvrez l'invite de commande dans le dossier contenant le dossier "img". Une fois que vous avez lancé la commande, FFMPEG traitera les images pour créer l'animation souhaitée.
