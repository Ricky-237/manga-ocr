# Plan d'entraînement OCR manga/webtoon mobile

## Summary
- Construire un OCR offline Android depuis zéro autour d’un pipeline `détection de lignes -> routage langue/script -> reconnaissance -> ordre de lecture`, optimisé pour `JP + KR + Latin`.
- Le périmètre v1 couvre les pages de manga et les tiles de webtoon, avec priorité aux dialogues et captions lisibles; les SFX très stylisés et le ruby/furigana minuscule passent en phase 2.
- Cibles de sortie: `CER end-to-end <= 2.5%` sur le benchmark dialogue-first, `recall détection >= 97%`, package total `<= 20 MB` en `INT8`, latence `<= 800 ms médiane` et `<= 1.2 s p95` par page 720p sur Android Go class (A53/A55, 2-4 GB RAM).

## Key Changes
- Définir un corpus `gold` de `20k` images/tiles annotées (`10k manga`, `10k webtoon`) avec `polygones de ligne`, `transcription`, `langue`, `direction`, `type de texte`, et séparation stricte par série/auteur entre train/val/test.
- Générer un corpus synthétique de `4M` crops de lignes et `400k` pages/tiles avec polices manga/webtoon, mise en page verticale/horizontale, bulles, screentones, moiré, compression, blur, contraste faible, et `5%` d’exposition contrôlée aux SFX.
- Constituer un corpus `silver` de `150k` pages/tiles sous licence ou domaine public, pseudo-annotées par un teacher ensemble et filtrées par consensus/confidence avant réentraînement.
- Utiliser un teacher offline non mobile: `DBNet++-ResNet34` pour la détection et `PARSeq-base` séparé `JP` / `KR+Latin` pour produire pseudo-labels et distillation.
- Utiliser un student mobile pour la détection: `MobileNetV3-Large + FPN + DB head`, avec sortie `polygones de ligne + direction verticale/horizontale`.
- Utiliser un routeur léger de script sur crop (`MobileNetV3-Small 0.35x`) pour choisir entre deux recognizers mobiles.
- Utiliser deux recognizers students `MobileNetV3-Small + temporal depthwise conv + CTC`, l’un pour `JP+Latin+ponctuation` avec vocabulaire cible `5k-6k`, l’autre pour `KR+Latin+ponctuation` avec vocabulaire cible `2k-3k`.
- Entraîner selon une séquence fixe: `prétrain synthétique -> fine-tune gold -> self-training silver -> distillation teacher/student -> pruning structuré 20% -> quantization-aware training INT8`.
- Exécuter la détection sur image normalisée avec côté long `1280 px`; pour les webtoons, découper en tiles `1280x1280` avec `10%` d’overlap puis dédupliquer par IoU des polygones.
- Calculer l’ordre de lecture par règles déterministes: webtoon `haut -> bas`; manga `clusters de lignes`, puis blocs verticaux `droite -> gauche` et `haut -> bas`, blocs horizontaux `haut -> bas` puis `gauche -> droite`.

## Interfaces
- Standardiser un `dataset_manifest.jsonl` avec les champs `image_path`, `series_id`, `domain`, `tile_id`, `polygon`, `transcript`, `lang`, `direction`, `text_type`, `split`.
- Standardiser un `train_ocr.yaml` avec sections `data`, `detector`, `router`, `recognizer_jp`, `recognizer_kr`, `distillation`, `pruning`, `qat`, `benchmarks`.
- Exposer côté mobile une API unique `runPage(bitmap)` retournant une liste de `OcrBlock { polygon, text, lang, direction, confidence, order_index }`.
- Exporter uniquement les students en `TFLite INT8`, avec `NNAPI` en priorité et fallback CPU; interdire toute couche non compatible TFLite/NNAPI dès la conception.

## Test Plan
- Mesurer séparément détection, reconnaissance et end-to-end sur `manga JP vertical`, `manga JP horizontal`, `webtoon KR horizontal`, `texte mixte Latin`.
- Garder des slices dédiées `petit texte`, `contraste faible`, `JPEG fort`, `scan bruité`, `long webtoon`, `bulles sur screentone`, `dialogues mixtes JP/KR/Latin`.
- Vérifier la parité `FP32 vs INT8` avec dérive maximale `<= 0.4 point` de CER absolu.
- Benchmarker sur au moins deux appareils Android bas de gamme réels et suivre `latence`, `RAM pic`, `taille du package`, `température/throttling`.
- Ajouter une suite d’acceptation phase 2 pour `SFX stylisés` et `furigana tiny`, sans les rendre bloquants pour la release v1.

## Assumptions
- Le projet démarre sans code existant, donc l’implémentation peut être structurée directement autour des artefacts `manifest/config/models/mobile API`.
- Les données réelles utilisées pour gold/silver seront sous licence ou domaine public; le plan ne suppose pas de scraping non autorisé.
- Android offline est la cible initiale; iOS viendra ensuite via un export/runtime séparé, sans changer l’architecture d’entraînement.
- La v1 vise la meilleure qualité utile en lecture/traduction; les SFX très artistiques et le ruby/furigana extrême sont explicitement planifiés en deuxième vague.
