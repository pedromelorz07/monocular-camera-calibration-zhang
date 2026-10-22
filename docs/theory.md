# Fundamentos matemáticos

## Projeção

Um ponto do tabuleiro em coordenadas homogêneas `X=[X,Y,Z,1]ᵀ` transforma-se em coordenadas de câmera por `Xc=RX+t`. A projeção ideal pinhole é `s[u,v,1]ᵀ=K[R|t]X`, com `K=[[fx,γ,cx],[0,fy,cy],[0,0,1]]`. O fator `s` é proporcional à profundidade. `fx` e `fy` são focais em pixels, `(cx,cy)` é o ponto principal e `γ` é skew entre eixos. O uso padrão do OpenCV assume skew nulo.

## Distorção

Após dividir por profundidade, coordenadas normalizadas `(x,y)` recebem distorção radial e tangencial. Com `r²=x²+y²`, `L=1+k1r²+k2r⁴+k3r⁶`, `xd=xL+2p1xy+p2(r²+2x²)` e `yd=yL+p1(r²+2y²)+2p2xy`. Os coeficientes são guardados na ordem OpenCV `[k1,k2,p1,p2,k3]`. O vetor de cinco parâmetros não deve ser confundido com os modelos de oito ou quatorze parâmetros nem com o modelo fisheye.

## Homografia e restrições de Zhang

Para um alvo no plano `Z=0`, cada pose define `H=K[r1 r2 t]`, onde `r1,r2` são as primeiras colunas da rotação. Defina `B=K⁻ᵀK⁻¹`, simétrica. Escrevendo `hi` como colunas de `H`, a ortogonalidade e igualdade de norma das colunas de rotação implicam `h1ᵀBh2=0` e `h1ᵀBh1−h2ᵀBh2=0`. Vistas em orientações diferentes fornecem um sistema para estimar os elementos independentes de `B`, recuperar `K` e então decompor cada `H` para obter a pose. A diversidade de inclinação é importante porque poses quase iguais tornam o sistema mal condicionado.

Na prática, `cv2.calibrateCamera()` usa correspondências 3D/2D e opções de calibração do OpenCV para estimar e refinar os parâmetros, incluindo minimização não linear do erro de reprojeção. O valor RMS retornado é relatado separadamente das estatísticas adicionais calculadas aqui.

## Erros

Para imagem `i` e canto `j`, `eij=||pij−p̂ij||₂` em pixels. A média por imagem é `mean_j(eij)`; a média global por ponto é `mean_ij(eij)`. O RMS por coordenada observado é `sqrt(mean_ij((duij²+dvij²)))`; máximo, mediana e percentil 95 são quantis/distâncias entre cantos. Um RMS por coordenada não equivale à média das normas euclidianas.

## Pose e coordenadas

Os vetores Rodrigues e translação estimados por imagem transformam pontos do referencial do tabuleiro para o da câmera: `Xc=R Xboard+t`. As unidades de `t` correspondem à unidade fornecida para o lado do quadrado. Ao usar os intrínsecos na odometria, a resolução deve coincidir com a da calibração ou a matriz deve ser escalada corretamente; o modelo óptico e o foco também precisam permanecer estáveis.
