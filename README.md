# simulation-onde
Je présente ici un code Python renvoyant une animation d'ondes.

Nous pouvons choisir à notre guise les conditions aux bords (périodiques, d'absorption ou réflectives) et le système d'étude (fente unique, double fente, lentille convergente, accélération de la source ou pluies) en enlevant les "#" des appels de fonctions depuis la fonction update et depuis les fonctions de célérité.

Une fois le code lancé et les calculs terminés, le code renvoie chaque image de l'animation (30FPS, que l'on peut modifier depuis ) d'une durée de 20 secondes (que l'on peut modifier via la variable tf) dans le dossier "img" (à créer prélablement).

Afin de créer une animation à partir de ces images, vous devez disposer du logiciel FFMPEG sur votre ordinateur et exécuter la commande mentionnée en bas du code (en tant que commentaire). Pour ce faire, ouvrez l'invite de commande dans le dossier contenant le dossier "img". Une fois que vous avez lancé la commande, FFMPEG traitera les images pour créer l'animation souhaitée.
