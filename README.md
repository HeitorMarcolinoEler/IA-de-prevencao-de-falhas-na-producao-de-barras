
# Credenciais SSH para VPS:

- Host: vps51980.publiccloud.com.br
- Porta: 22
- Usuário: root
- Senha: Falhas@#$


# IDEAS

## Definição de um novo ROI com base no framerate

> Definir um novo ROI dinamico (ROI estático -> novo ROI -> ROI dinamico barra). <br>
> Com base no framerate, irá definir um ROI mais ou menos largo. <br>
> Isso porque a largura do ROI está totalmente atrelada a qualidade do vídeo de mostrar os detalhes. <br>
> As vantagens de ter um ROI menor seria que filtrando uma área menor, caso tiver ruídos, avarías, irá aumentar a porcentagem consideravelmente, do que filtrando uma área maior. <br>
> Precisa fazer função de monitoramento de framerate automática.

- Exemplos: <br>
Framerate = 60fps -> Largura_ROI = 20p <br>
Framerate = 40fps -> Largura_ROI = 50p <br>
Framerate = 30fps -> Largura_ROI = 90p <br>
Framerate = 25fps -> Largura_ROI = 110p <br>
Framerate = 22fps -> Largura_ROI = 150p <br>
Framerate = 19fps -> Largura_ROI = 190p <br>

### Possiveis problemas

> Pode acontecer instabilidades em gerar um ROI dinamico caso a largura do ROI for muito pequena, principalmente se acontecer muito ruído de uma vez na imagem, caso acontecer é interessante estudar
> outra forma de estrutura de ROI Ex: (ROI estático -> ROI dinamico barra (atual) -> Novo ROI)
