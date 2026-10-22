# Metodologia experimental

## Experimento sintético

O gerador fixa uma matriz intrínseca e vetor de distorção conhecidos, gera vetores de pose reprodutíveis por semente e projeta os quatro cantos externos do tabuleiro. Uma homografia renderiza a textura plana numa imagem ideal; um mapa inverso baseado em `undistortPoints` remapeia a imagem para gerar a observação distorcida. A interpolação bilinear e a amostragem raster introduzem diferenças em relação à geometria contínua. A referência e as poses são gravadas separadamente das imagens e nunca são lidas pelo calibrador.

## Detecção e estimação

Para cada imagem colorida carregada, converte-se para cinza, detectam-se os cantos internos com `findChessboardCorners` e refina-se com `cornerSubPix`. Vistas sem detecção são reportadas. As imagens devem ter resolução consistente e o mínimo de vistas configurado. Cada canto 2D corresponde ao ponto 3D coplanar construído com passo físico conhecido. A calibração chama `cv2.calibrateCamera()` e registra matriz, coeficientes, pose por vista e RMS retornado.

## Avaliação

Os pontos observados e reprojetados via `cv2.projectPoints()` alimentam as métricas em `metrics.py`. O relatório define as agregações por imagem e global e inclui distâncias individuais para histograma. Para avaliar sintético, compare a estimativa com `synthetic_ground_truth.json`; diferenças são esperadas por interpolação, quantização, ruído de detecção e correlação dos parâmetros. Os valores do ground truth são exclusivamente referência de avaliação.

## Arquivos e limites

O YAML armazena os parâmetros estimados usados para corrigir imagens, a resolução e o modelo `plumb_bob`. O relatório guarda parâmetros e poses; as figuras mostram cantos e resíduos. Resultados sintéticos e reais devem permanecer identificados em pastas distintas. O método assume padrão plano medido com precisão, lente aproximadamente pinhole, foco fixo, padrão nítido e variedade suficiente de poses. Variações de resolução, foco ou zoom invalidam a aplicação direta dos parâmetros sem ajuste.
