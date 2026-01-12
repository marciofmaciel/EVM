`markdown
🔬 Aplicativo EVM para Análise de Tensões Residuais - Documentação Completa

![Python](https://www.python.org/)
![Streamlit](https://streamlit.io/)
![Computer Vision](https://opencv.org/)
![Signal Processing](https://scipy.org/)
![License](LICENSE)

---

1. INTRODUÇÃO 📚

Este documento serve como um guia abrangente para o Aplicativo EVM para Análise de Tensões Residuais, uma ferramenta desenvolvida em Python utilizando o framework Streamlit. O projeto tem como objetivo principal aplicar a técnica de Eulerian Video Magnification (EVM) em vídeos de entrada para amplificar variações temporais sutis, que são frequentemente invisíveis a olho nu, e correlacioná-las com um índice qualitativo de tensões residuais em estruturas.

A motivação científica por trás desta aplicação reside na premissa de que tensões residuais em materiais e estruturas alteram suas propriedades mecânicas, como a rigidez local. Essas alterações, por sua vez, influenciam a resposta vibracional da estrutura, modificando suas frequências naturais e modos de vibração. Ao amplificar e analisar essas vibrações sutis, podemos inferir regiões com maior ou menor energia de resposta modal, que podem ser indicativas de concentrações de tensão ou variações de rigidez.

O aplicativo oferece uma interface gráfica intuitiva que permite aos usuários carregar vídeos, configurar parâmetros de processamento EVM (como bandas de frequência e ganho de amplificação), e visualizar os resultados na forma de um "mapa de calor" sobreposto ao vídeo original. Além disso, fornece ferramentas para exportar dados e imagens para análises posteriores. Este projeto é uma ponte entre a visão computacional avançada e a engenharia mecânica, oferecendo uma nova perspectiva para a inspeção não destrutiva e a análise estrutural.

> ⚠️ AVISO CRÍTICO:
> O resultado gerado por este aplicativo é um índice RELATIVO associado à resposta vibracional da estrutura. Os valores apresentados NÃO são tensões absolutas (em unidades como MPa ou Pa) e não devem ser utilizados como substitutos de ensaios destrutivos, ensaios não destrutivos (END) certificados ou análises mecânicas quantitativas. A interpretação dos resultados requer conhecimento técnico e validação externa.

🌟 Principais Funcionalidades:

*   Upload de Vídeo Flexível: Suporte a formatos MP4/AVI com detecção automática de FPS e dimensões.
*   Processamento EVM Avançado: Aplicação de filtros temporais band-pass (Butterworth) e amplificação de variações de luminância por pixel.
*   Métrica de Tensão Relativa: Cálculo do RMS temporal do sinal filtrado por pixel como proxy para a energia vibracional.
*   Normalização Robusta: Utilização de percentis (p5-p95) para normalização da métrica, reduzindo o impacto de outliers.
*   Geração de Heatmap Dinâmico: Overlay de mapa de calor configurável (colormap, opacidade) sobre o vídeo original.
*   Controles Interativos: Sidebar com sliders e campos para ajuste de todos os parâmetros de processamento e visualização.
*   ROI Opcional: Definição de Região de Interesse para focar a análise em áreas específicas.
*   Exportação de Resultados: Download do vídeo processado, imagem estática do heatmap, e dados CSV com métricas por pixel.
*   Pré-visualização em Tempo Real: Amostra de frames e barra de progresso para monitoramento do processamento.
*   Disclaimers Integrados: Mensagens claras sobre as limitações e melhores práticas para o uso da ferramenta.

🎯 Público-Alvo:

Este aplicativo é destinado a engenheiros, pesquisadores, estudantes e profissionais das áreas de engenharia mecânica, civil, materiais e visão computacional que buscam uma ferramenta exploratória e qualitativa para análise de vibrações e detecção de padrões em estruturas.

💡 Casos de Uso Resumidos:

*   Identificação de regiões com maior amplitude de vibração em estruturas.
*   Análise qualitativa de integridade estrutural.
*   Detecção de anomalias vibracionais em componentes mecânicos.
*   Estudo do comportamento dinâmico de materiais.
*   Ferramenta educacional para demonstração de conceitos de EVM e análise modal.

---

2. FUNDAMENTOS TEÓRICOS 📚

2.1 O que é Eulerian Video Magnification (EVM)

O Eulerian Video Magnification (EVM) é uma técnica de processamento de vídeo que permite revelar e amplificar pequenas variações temporais em um vídeo que são imperceptíveis a olho nu. Essas variações podem ser movimentos sutis, mudanças de cor ou variações de intensidade de luz. A técnica foi introduzida por Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Frédo Durand e William T. Freeman em 2012, em seu artigo seminal "Eulerian Video Magnification for Revealing Subtle Changes in the World".

O conceito central do EVM é tratar o vídeo como uma função contínua de espaço e tempo, onde cada pixel (ou região de pixels) possui um sinal temporal associado. Em vez de rastrear o movimento de objetos (abordagem Lagrangiana), o EVM observa as mudanças em pontos fixos no espaço (abordagem Euleriana). Isso permite que variações minúsculas em cada pixel sejam isoladas, filtradas e amplificadas.

*   Abordagem Euleriana: Foca nas mudanças que ocorrem em um ponto fixo no espaço ao longo do tempo. É como observar um sensor em cada pixel da imagem e registrar seu sinal.
*   Abordagem Lagrangiana: Foca no rastreamento do movimento de objetos ou pontos específicos no espaço ao longo do tempo.

A vantagem da abordagem Euleriana é que ela não requer detecção ou rastreamento de características, tornando-a robusta para movimentos complexos e variações de pequena escala.

2.2 Pipeline de Processamento EVM

O processo de EVM envolve uma série de etapas para isolar, filtrar e amplificar as variações temporais:

1.  Decomposição Espacial (Pirâmide Laplaciana)
    *   Objetivo: Separar o vídeo em diferentes bandas de frequência espacial (detalhes finos, médios e grosseiros). Isso é crucial porque variações sutis podem ocorrer em diferentes escalas espaciais.
    *   Como funciona: Uma pirâmide Laplaciana é construída para cada frame do vídeo. Primeiro, uma pirâmide Gaussiana é criada por sucessivas aplicações de filtros Gaussianos e subamostragem. A pirâmide Laplaciana é então formada pela diferença entre cada nível da pirâmide Gaussiana e a versão expandida do nível seguinte. Isso resulta em uma representação multi-escala onde cada nível contém os detalhes de uma banda de frequência espacial específica.
    *   Benefício: Permite que a amplificação seja aplicada seletivamente em diferentes escalas, melhorando a relação sinal-ruído (SNR) para variações de baixa amplitude e evitando artefatos em detalhes grosseiros.

2.  Filtragem Temporal (Band-Pass por Pixel)
    *   Objetivo: Isolar as variações temporais de interesse dentro de uma banda de frequência específica e remover ruídos ou movimentos indesejados fora dessa banda.
    *   Como funciona: Para cada pixel (ou coeficiente em cada nível da pirâmide Laplaciana), o sinal de intensidade ao longo do tempo é tratado como uma série temporal. Um filtro passa-banda (geralmente um filtro Butterworth) é aplicado a esta série temporal. Este filtro permite a passagem de frequências entre f_low e f_high, atenuando as frequências abaixo de f_low e acima de f_high.
    *   Benefício: Permite focar em vibrações que ocorrem em uma faixa de frequência específica, como as frequências naturais de uma estrutura, ignorando movimentos de baixa frequência (e.g., movimento de câmera lento) ou ruídos de alta frequência.

3.  Amplificação (Ganho Alpha)
    *   Objetivo: Aumentar a magnitude das variações temporais filtradas.
    *   Como funciona: O sinal temporal filtrado (δ_filt) é multiplicado por um fator de ganho α. Este fator determina o quão intensamente as variações serão amplificadas.
    *   Benefício: Torna as variações sutis visíveis.
    *   Cuidado: Um α muito alto pode introduzir artefatos, como saturação de cor, ruído amplificado ou distorções visuais.

4.  Reconstrução
    *   Objetivo: Recompor o vídeo amplificado a partir dos níveis da pirâmide Laplaciana amplificados e adicionar essas variações ao vídeo original.
    *   Como funciona: Os níveis da pirâmide Laplaciana amplificados são somados de volta, começando do nível mais grosseiro e expandindo cada nível antes de somá-lo ao próximo. O resultado final é adicionado ao frame original (ou ao frame base da pirâmide Gaussiana) para gerar o frame amplificado.
    *   Benefício: Produz um vídeo onde as variações sutis são visivelmente exageradas, mantendo a estrutura geral do vídeo original.

2.3 Matemática do EVM

Vamos formalizar as etapas com algumas equações. Considere I(x, y, t) a intensidade de um pixel na posição (x, y) no tempo t.

*   Sinal Base: A intensidade de um pixel pode ser vista como a soma de uma componente estática (ou de baixa frequência) e uma componente de variação temporal sutil.
    $$ I(x, y, t) = f(x, y) + \delta(x, y, t) $$
    Onde:
    *   f(x, y) é a intensidade base ou média do pixel (componente estática/lenta).
    *   δ(x, y, t) é a variação temporal sutil que queremos amplificar.

*   Sinal Filtrado: Para cada pixel (x, y), o sinal δ(x, y, t) ao longo do tempo t é submetido a um filtro passa-banda B.
    $$ \delta_{filt}(x, y, t) = B(\delta(x, y, t)) $$
    Onde:
    *   B representa a operação do filtro passa-banda (e.g., Butterworth).

*   Sinal Amplificado: O sinal filtrado é então amplificado por um fator α.
    $$ I_{amp}(x, y, t) = f(x, y) + \alpha \cdot \delta_{filt}(x, y, t) $$
    Onde:
    *   α é o fator de ganho (alpha).

*   Métrica RMS (Root Mean Square): Para quantificar a intensidade da resposta vibracional em cada pixel, calculamos o valor RMS do sinal filtrado ao longo do tempo. Isso nos dá uma medida da "energia" média da vibração em cada ponto.
    $$ A_{RMS}(x, y) = \sqrt{\frac{1}{T} \int_{0}^{T} \delta_{filt}^2(x, y, t) dt} $$
    Onde:
    *   T é a duração total do vídeo ou do segmento de tempo analisado.

*   Normalização Robusta: Para criar um mapa de calor visualmente significativo e robusto a outliers, a métrica A_RMS é normalizada usando percentis. Isso mapeia os valores para uma faixa de 0 a 1.
    $$ A_{norm}(x, y) = \frac{A_{RMS}(x, y) - P_5}{P_{95} - P_5} $$
    Onde:
    *   P_5 é o 5º percentil dos valores de A_RMS em toda a imagem.
    *   P_{95} é o 95º percentil dos valores de A_RMS em toda a imagem.
    *   Os valores resultantes são então "clamped" (limitados) entre 0 e 1.

2.4 Conexão com Tensões Residuais

A análise de tensões residuais é um campo crítico na engenharia de materiais e estruturas. Tensões residuais são tensões que permanecem em um material ou estrutura na ausência de cargas externas. Elas podem ser introduzidas por processos de fabricação (soldagem, conformação, tratamento térmico) ou por danos (fadiga, corrosão).

*   Como Tensões Residuais Afetam a Vibração:
    *   Modificação da Rigidez: A presença de tensões residuais pode alterar a rigidez efetiva de uma seção do material. Tensões de compressão tendem a aumentar a rigidez, enquanto tensões de tração podem diminuí-la (especialmente em casos de trincas ou danos).
    *   Frequências Naturais: A frequência natural de vibração de uma estrutura é diretamente relacionada à sua rigidez e massa (ω = √(k/m)). Se a rigidez local é alterada por tensões residuais, as frequências naturais de vibração daquela região também serão modificadas.
    *   Modos de Vibração: Os modos de vibração (padrões de deformação que uma estrutura assume quando vibra em uma frequência natural) também são influenciados pela distribuição de rigidez. Regiões com tensões residuais podem apresentar modos de vibração diferentes ou com maior/menor amplitude de resposta.
    *   Distribuição de Energia na Resposta Modal: Regiões com maior concentração de tensões residuais (ou com danos associados a elas) podem exibir uma resposta vibracional mais intensa ou em frequências diferentes quando a estrutura é excitada. O mapa de calor gerado pelo EVM, que representa a energia RMS do sinal filtrado, pode, portanto, indicar qualitativamente essas regiões de interesse.

2.5 Limitações Teóricas

Embora o EVM seja uma ferramenta poderosa, é crucial entender suas limitações teóricas e práticas:

1.  Natureza Qualitativa: A principal limitação é que o EVM, por si só, não fornece valores quantitativos de tensão (e.g., em MPa). Ele amplifica variações de intensidade de pixel, que são um proxy para deslocamentos ou deformações. A correlação direta com tensões absolutas exigiria modelos mecânicos complexos, calibração com dados de ensaios e conhecimento das propriedades do material. O mapa de calor é um índice relativo de resposta vibracional.

2.  Sensibilidade a Ruído: O EVM amplifica todas as variações temporais dentro da banda de frequência definida, incluindo ruído. Ruído de câmera (sensor), ruído de quantização, e ruído ambiental (e.g., vibrações de outras fontes) podem ser amplificados, levando a artefatos. A qualidade do vídeo de entrada é, portanto, fundamental.

3.  Dependência da Iluminação: Variações na iluminação ambiente são interpretadas como variações de intensidade de pixel e serão amplificadas. Isso pode mascarar as vibrações reais ou criar artefatos significativos. Iluminação constante e uniforme é essencial.

4.  Movimento de Câmera: Qualquer movimento da câmera, mesmo que sutil, será amplificado. Isso pode ser confundido com movimento da estrutura ou gerar artefatos de "ondas" no vídeo. O uso de um tripé robusto e estabilização de imagem (se disponível) é mandatório.

5.  Critério de Nyquist: A frequência de amostragem temporal (FPS do vídeo) impõe um limite superior às frequências que podem ser detectadas e amplificadas. A frequência máxima detectável é a frequência de Nyquist (FPS/2). Se a vibração de interesse ocorrer acima dessa frequência, ela não será capturada corretamente, podendo levar a aliasing.

6.  Artefatos de Amplificação: Um fator de ganho α muito alto pode levar a artefatos visuais como saturação de pixels (cores estouradas), distorções geométricas ou "ondas" no vídeo, especialmente em regiões com movimento já visível ou alto ruído.

7.  Assunção de Pequenos Movimentos: O EVM funciona melhor para movimentos de pequena amplitude. Para movimentos grandes, as aproximações lineares usadas na técnica podem não ser válidas, levando a distorções.

---

3. ARQUITETURA DO SISTEMA 🏗️

3.1 Visão Geral

O aplicativo é construído sobre o framework Streamlit, que fornece a interface de usuário (UI) e gerencia o fluxo de dados. O processamento central é realizado em Python, utilizando bibliotecas otimizadas para visão computacional e processamento de sinais.

Fluxo de Dados (Textual):

1.  Entrada de Vídeo: O usuário faz upload de um arquivo de vídeo (MP4/AVI) através da interface Streamlit.
2.  Leitura e Pré-processamento: O vídeo é lido frame a frame. Cada frame é convertido para escala de cinza (ou canal de luminância) e normalizado.
3.  Empilhamento de Frames: Os frames pré-processados são empilhados em uma estrutura de dados (tensor (T, H, W)) para facilitar o processamento temporal por pixel.
4.  Processamento EVM:
    *   Filtragem Temporal: Para cada pixel (x, y), o sinal de intensidade ao longo do tempo t é extraído e um filtro passa-banda (Butterworth) é aplicado.
    *   Cálculo RMS: O valor RMS do sinal filtrado é calculado para cada pixel, resultando em um mapa de A_RMS(x,y).
5.  Normalização e Heatmap: O mapa A_RMS é normalizado usando percentis e mapeado para um colormap para gerar o heatmap.
6.  Geração de Vídeo de Saída: Cada frame original do vídeo é combinado com o heatmap (overlay com transparência ajustável). Os frames resultantes são então gravados em um novo arquivo de vídeo.
7.  Exportação de Dados: O mapa de calor estático, as estatísticas e os dados brutos por pixel são exportados em formatos PNG e CSV.
8.  Interface do Usuário: A UI do Streamlit exibe pré-visualizações, barras de progresso e botões de download para os resultados.

3.2 Módulos Principais

*   Módulo de Entrada de Vídeo (load_video): Responsável por carregar o arquivo de vídeo, extrair frames, determinar FPS e dimensões. Lida com a conversão para escala de cinza e normalização inicial.
*   Módulo de Processamento EVM (apply_temporal_bandpass): Contém a lógica central para aplicar o filtro passa-banda Butterworth a cada série temporal de pixel.
*   Módulo de Cálculo de Métrica (compute_rms_map): Calcula o valor RMS do sinal filtrado para cada pixel.
*   Módulo de Normalização (normalize_map): Aplica a normalização robusta por percentis ao mapa RMS.
*   Módulo de Geração de Heatmap (generate_heatmap_overlay): Mapeia os valores normalizados para um colormap e cria a imagem do heatmap, aplicando-a como overlay sobre os frames originais.
*   Módulo de Exportação de Saída (write_output_video, export_data): Gerencia a escrita do vídeo final e a exportação de imagens e dados CSV.
*   Módulo de Interface do Usuário (Streamlit): Orquestra todos os módulos, gerencia a interação do usuário, exibe controles, pré-visualizações e resultados.

3.3 Tecnologias Utilizadas

| Streamlit       | 1.31.0      | Framework para construção da interface web interativa                  |
| SciPy           | 1.11.4      | Processamento de sinais (filtros Butterworth, sosfiltfilt)          |
| Matplotlib      | 3.8.2       | Geração de mapas de calor e colormaps                                  |
| Pillow (PIL)    | 10.2.0      | Manipulação de imagens (usado indiretamente por outras libs)           |
| Scikit-image    | (opcional)  | Para implementação de pirâmides Laplacianas (não usado na implementação principal) |

---

4. REQUISITOS E INSTALAÇÃO ⚙️

4.1 Requisitos de Hardware

*   CPU: Processador multi-core (Intel Core i5/Ryzen 5 ou superior) é altamente recomendado para processamento de vídeo.
*   RAM: Mínimo de 4 GB, mas 8 GB ou mais são fortemente recomendados para vídeos de maior resolução e duração, a fim de evitar erros de memória.
*   Armazenamento: Espaço em disco suficiente para armazenar vídeos de entrada e saída (pode ser significativo para vídeos longos).

4.2 Requisitos de Software

*   Sistema Operacional: Windows 10/11, macOS (Monterey ou superior), Linux (Ubuntu 20.04+ ou distribuições equivalentes).
*   Python: Versão 3.10 ou superior.

4.3 Instalação Passo a Passo

Método 1: Instalação Padrão (Recomendado)

1.  Crie a pasta do projeto:
    `bash
    mkdir evm-stress-analysis
    cd evm-stress-analysis
    `

2.  Salve os arquivos:
    Certifique-se de que os seguintes arquivos estejam na pasta evm-stress-analysis/:
    *   streamlit_app.py
    *   generate_synthetic_test_video.py
    *   requirements.txt
    *   README.md (este arquivo)

3.  Crie e ative um ambiente virtual (altamente recomendado para isolar as dependências do projeto):
    `bash
    python -m venv venv
    `
    *   No Windows:
        `bash
        venv\Scripts\activate
        `
    *   No Linux/macOS:
        `bash
        source venv/bin/activate
        `

4.  Instale as dependências:
    `bash
    pip install -r requirements.txt
    `

5.  Verificação da Instalação:
    Após a instalação, você pode verificar se as bibliotecas foram instaladas corretamente:
    `bash
    pip list
    `
    Você deverá ver streamlit, numpy, scipy, opencv-python-headless, matplotlib, pandas e pillow na lista.

Método 2: Instalação com Conda (Alternativo)

Se você usa Anaconda ou Miniconda, pode criar um ambiente Conda:

1.  Crie o ambiente Conda:
    `bash
    conda create -n evm_env python=3.10
    `

2.  Ative o ambiente:
    `bash
    conda activate evm_env
    `

3.  Instale as dependências:
    `bash
    pip install -r requirements.txt
    `

Método 3: Docker (Opcional - para ambientes isolados)

Para uma instalação totalmente isolada e reprodutível, você pode usar Docker. Crie um arquivo Dockerfile na raiz do projeto:

`dockerfile
Use uma imagem base Python
FROM python:3.10-slim-buster

Defina o diretório de trabalho
WORKDIR /app

Copie o arquivo de requisitos e instale as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

Copie o restante do código da aplicação
COPY . .

Exponha a porta que o Streamlit usa
EXPOSE 8501

Comando para iniciar a aplicação Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
`

1.  Construa a imagem Docker:
    `bash
    docker build -t evm-app .
    `

2.  Execute o contêiner Docker:
    `bash
    docker run -p 8501:8501 evm-app
    `
    A aplicação estará acessível em http://localhost:8501.

4.4 Verificação da Instalação

Após instalar as dependências (Método 1 ou 2), você pode verificar se o Streamlit está funcionando:

`bash
streamlit hello
`
Isso deve abrir uma aplicação de demonstração do Streamlit no seu navegador. Se funcionar, sua instalação base está correta.

---

5. GUIA DE USO COMPLETO 🚀

5.1 Iniciando a Aplicação

1.  Ative seu ambiente virtual (se estiver usando um):
    *   Windows: venv\Scripts\activate
    *   Linux/macOS: source venv/bin/activate

2.  Execute o script principal:
    `bash
    streamlit run streamlit_app.py
    `

3.  Saída Esperada: Seu navegador padrão deve abrir automaticamente em http://localhost:8501, exibindo a interface do aplicativo. Se não abrir, copie e cole o endereço no seu navegador.

5.2 Interface do Usuário

A interface do aplicativo é dividida em duas áreas principais:

*   Sidebar (Barra Lateral): Localizada à esquerda, contém todos os controles e parâmetros para o processamento EVM, normalização, visualização e opções de performance.
*   Área Principal: Ocupa a maior parte da tela, exibindo a zona de upload de vídeo, pré-visualizações, barras de progresso, resultados (vídeo processado, heatmap) e botões de download.

Detalhes da Interface:

*   Zona de Upload: Na área principal, um componente st.file_uploader permite arrastar e soltar ou selecionar um arquivo de vídeo.
*   Pré-visualização do Vídeo Original: Após o upload, um frame de amostra do vídeo original é exibido.
*   Parâmetros EVM (Sidebar):
    *   Frequência baixa (Hz): Slider para definir o limite inferior do filtro passa-banda.
    *   Frequência alta (Hz): Slider para definir o limite superior do filtro passa-banda.
    *   Ganho Alpha: Slider para ajustar o fator de amplificação.
    *   Ordem do filtro: Slider para definir a ordem do filtro Butterworth.
    *   FPS do Vídeo (Detectado): Campo informativo que mostra o FPS detectado. Se for 0, o usuário pode inserir manualmente.
*   Parâmetros de Normalização (Sidebar):
    *   Percentil baixo (p5): Slider para o percentil inferior da normalização.
    *   Percentil alto (p95): Slider para o percentil superior da normalização.
*   Parâmetros de Visualização (Sidebar):
    *   Colormap: Dropdown para selecionar o esquema de cores do heatmap.
    *   Opacidade do Overlay: Slider para ajustar a transparência do heatmap sobre o vídeo original.
*   Parâmetros de Performance (Sidebar):
    *   Máximo de frames para preview: Slider para limitar o número de frames processados para a pré-visualização, útil para vídeos muito longos.
*   ROI (Região de Interesse - Sidebar):
    *   Campos numéricos para X_min, Y_min, X_max, Y_max para definir uma área retangular de interesse.
*   Botão "Processar Vídeo": Inicia o pipeline de processamento EVM.
*   Barra de Progresso: Exibida durante o processamento.
*   Área de Resultados: Após o processamento, exibe o vídeo com overlay, o heatmap estático e os botões de download.
*   Botões de Download: Permitem baixar o vídeo processado, o heatmap em PNG, as estatísticas em CSV e os dados RMS por pixel em CSV.

5.3 Workflow Básico

Siga estes passos para processar um vídeo:

1.  Inicie a aplicação Streamlit conforme descrito em 5.1.
2.  Faça o upload do seu vídeo: Clique em "Faça upload do vídeo" na área principal ou arraste e solte o arquivo.
3.  Verifique o FPS: O aplicativo tentará detectar o FPS. Se o valor for 0 ou incorreto, insira o FPS correto manualmente no campo "FPS do Vídeo (Detectado)".
4.  Pré-visualize o Frame: Um frame do vídeo original será exibido.
5.  Configure os Parâmetros EVM (Sidebar):
    *   Ajuste Frequência baixa (Hz) e Frequência alta (Hz) para a faixa de frequência das vibrações que você deseja amplificar.
    *   Ajuste o Ganho Alpha para controlar a intensidade da amplificação. Comece com valores baixos (e.g., 10-20) e aumente gradualmente.
    *   Defina a Ordem do filtro (5 é um bom ponto de partida).
6.  Configure os Parâmetros de Normalização (Sidebar):
    *   Ajuste Percentil baixo (p5) e Percentil alto (p95) para controlar a faixa de normalização do heatmap. Valores padrão (5 e 95) são geralmente bons.
7.  Defina as Opções de Visualização (Sidebar):
    *   Escolha um Colormap que seja adequado para sua análise (e.g., inferno para percepção de gradientes, turbo para alto contraste).
    *   Ajuste a Opacidade do Overlay para equilibrar a visibilidade do heatmap e do vídeo original.
8.  Defina uma ROI (Opcional - Sidebar): Se desejar analisar apenas uma parte do vídeo, insira as coordenadas X_min, Y_min, X_max, Y_max.
9.  Clique em "Processar Vídeo": O processamento será iniciado. Uma barra de progresso será exibida.
10. Revise os Resultados: Após o processamento, o vídeo com o overlay do heatmap e uma imagem estática do heatmap serão exibidos.
11. Baixe os Resultados: Utilize os botões de download para salvar os arquivos gerados.

5.4 Gerando Vídeo Sintético para Validação

Para testar e validar a aplicação, é fornecido um script para gerar um vídeo sintético com vibração controlada:

1.  Certifique-se de que seu ambiente virtual está ativo.
2.  Execute o script:
    `bash
    python generate_synthetic_test_video.py
    `
3.  Saída Esperada:
    *   Uma pasta samples/ será criada na raiz do projeto (se não existir).
    *   Um arquivo synthetic_test_video.mp4 será salvo dentro da pasta samples/.
    *   O console exibirá mensagens de progresso e estatísticas finais do vídeo gerado.
    *   Este vídeo terá uma vibração senoidal de 3 Hz, com amplitude variando linearmente de 0 pixels (à esquerda) a 5 pixels (à direita), sobre um fundo de faixas horizontais.

4.  Use este vídeo na aplicação: Faça o upload de samples/synthetic_test_video.mp4 na interface do Streamlit e configure os parâmetros para f_low=0.5 Hz e f_high=3.0 Hz para observar o gradiente de vibração.

---

6. PARÂMETROS TÉCNICOS DETALHADOS 🎛️

6.1 Parâmetros EVM

Frequência baixa (f_low)
*   Descrição: Define o limite inferior da banda de frequência do filtro passa-banda. Apenas as variações temporais com frequência acima de f_low serão consideradas para amplificação.
*   Faixa de Valor: 0.1 Hz a FPS/2 - 0.1 Hz
*   Valores Recomendados: Depende da frequência de vibração da estrutura. Para vibrações lentas, comece em 0.5 Hz. Para vibrações mais rápidas, ajuste conforme necessário.
*   Efeito na Saída:
    *   Muito baixo: Pode incluir movimentos de câmera lentos ou ruído de baixa frequência.
    *   Muito alto: Pode cortar as vibrações de interesse se elas forem mais lentas que f_low.
*   Exemplo: Se a estrutura vibra a 5 Hz, f_low deve ser menor que 5 Hz (e.g., 4 Hz).

Frequência alta (f_high)
*   Descrição: Define o limite superior da banda de frequência do filtro passa-banda. Apenas as variações temporais com frequência abaixo de f_high serão consideradas para amplificação.
*   Faixa de Valor: f_low + 0.1 Hz a FPS/2 - 0.1 Hz
*   Valores Recomendados: Deve ser ligeiramente maior que a frequência de vibração de interesse.
*   Efeito na Saída:
    *   Muito baixo: Pode cortar as vibrações de interesse se elas forem mais rápidas que f_high.
    *   Muito alto: Pode incluir ruído de alta frequência ou aliasing se exceder a frequência de Nyquist (FPS/2).
*   Exemplo: Se a estrutura vibra a 5 Hz, f_high deve ser maior que 5 Hz (e.g., 6 Hz).

> ⚠️ Critério de Nyquist: É fundamental que f_high seja sempre menor que a metade do FPS do vídeo (FPS/2). Se f_high for igual ou maior que FPS/2, ocorrerá aliasing, e o filtro não funcionará corretamente.

Ganho Alpha
*   Descrição: Fator de amplificação aplicado ao sinal temporal filtrado. Controla a intensidade com que as variações sutis são exageradas.
*   Faixa de Valor: 1 a 500 (valores muito altos podem causar artefatos)
*   Valores Recomendados: Comece com 10-20 para movimentos sutis. Aumente gradualmente.
*   Efeito na Saída:
    *   Baixo: Pouca ou nenhuma amplificação visível.
    *   Alto: Variações sutis tornam-se muito visíveis, mas podem introduzir artefatos como saturação de cor, ruído amplificado ou distorções.
*   Exemplo: Para uma vibração quase imperceptível, alpha=50 pode ser necessário. Para movimentos já visíveis, alpha=5 pode ser suficiente.

Ordem do filtro
*   Descrição: Define a ordem do filtro Butterworth. A ordem do filtro afeta a inclinação da curva de atenuação (roll-off) nas bordas da banda passante.
*   Faixa de Valor: 1 a 10
*   Valores Recomendados: 4 a 6 são geralmente bons para um bom equilíbrio entre seletividade e suavidade.
*   Efeito na Saída:
    *   Baixa ordem: Transição mais suave entre as bandas, mas menos seletivo (pode deixar passar mais ruído).
    *   Alta ordem: Transição mais abrupta, mais seletivo, mas pode introduzir mais oscilações na resposta do filtro (ringing artifacts).

Nível da pirâmide (Não implementado diretamente nesta versão)
*   Descrição: Refere-se ao número de níveis na pirâmide Laplaciana. Controla a granularidade da decomposição espacial.
*   Recomendação: Em implementações completas de EVM, 3 a 5 níveis são comuns. Esta versão simplificada não utiliza pirâmide Laplaciana para focar no pipeline temporal, mas é uma melhoria futura.

6.2 Parâmetros de Normalização

Percentil baixo (p5)
*   Descrição: Define o percentil inferior para a normalização do mapa RMS. Valores de A_RMS abaixo deste percentil serão mapeados para 0 (cor mais fria).
*   Faixa de Valor: 0 a 49
*   Valores Recomendados: 5 (padrão)
*   Rationale: Ajuda a remover o ruído de fundo e os valores de A_RMS muito baixos que não representam vibração significativa, tornando o heatmap mais contrastado e focado nas regiões de interesse.

Percentil alto (p95)
*   Descrição: Define o percentil superior para a normalização do mapa RMS. Valores de A_RMS acima deste percentil serão mapeados para 1 (cor mais quente).
*   Faixa de Valor: 51 a 100
*   Valores Recomendados: 95 (padrão)
*   Rationale: Ajuda a evitar que outliers de alta amplitude (e.g., ruído pontual, movimento brusco) saturem o colormap, garantindo que a maior parte da faixa de valores seja utilizada para representar as variações significativas.

6.3 Parâmetros de Visualização

Colormap
*   Descrição: Esquema de cores utilizado para mapear os valores normalizados de A_norm para o heatmap.
*   Opções: inferno, turbo, viridis, plasma, magma, cividis, jet, hot, cool, gray, etc.
*   Recomendação:
    *   inferno, viridis, plasma: Perceptualmente uniformes, bons para visualização de dados científicos, evitam distorções de percepção.
    *   turbo: Alto contraste, bom para destacar diferenças.
    *   jet: Colormap clássico, mas pode introduzir artefatos visuais e não é perceptualmente uniforme.
*   Efeito: Altera a forma como os gradientes de tensão relativa são visualizados.

Opacidade do Overlay
*   Descrição: Controla a transparência do heatmap quando sobreposto ao vídeo original.
*   Faixa de Valor: 0.0 (totalmente transparente) a 1.0 (totalmente opaco).
*   Valores Recomendados: 0.5 a 0.7 para um bom equilíbrio entre a visibilidade do heatmap e do contexto do vídeo.
*   Efeito: Permite ver a estrutura original por baixo do heatmap.

ROI (Região de Interesse)
*   Descrição: Permite definir uma área retangular específica do frame para análise. Apenas os pixels dentro desta região serão processados e considerados para o cálculo do heatmap.
*   Parâmetros: X_min, Y_min, X_max, Y_max (coordenadas em pixels).
*   Efeito: Reduz o tempo de processamento e foca a análise em uma área específica, ignorando ruídos ou movimentos fora dela. Se não definido, o vídeo inteiro é processado.

6.4 Parâmetros de Performance

Máximo de frames para preview
*   Descrição: Limita o número de frames processados para a pré-visualização e para o cálculo do heatmap. Útil para vídeos muito longos onde o processamento completo levaria muito tempo.
*   Faixa de Valor: 10 a Todos os frames
*   Valores Recomendados: 100 a 300 para testes rápidos. Para análise completa, use "Todos os frames".
*   Efeito: Reduz o tempo de processamento, mas pode não capturar toda a dinâmica do vídeo.

Processamento por blocos (Não implementado diretamente nesta versão)
*   Descrição: Estratégia para processar vídeos muito longos ou de alta resolução dividindo-os em blocos temporais ou espaciais para gerenciar o uso de memória.
*   Recomendação: Para vídeos que excedem a memória RAM disponível, esta técnica é essencial. Esta versão processa o vídeo completo na memória, o que pode ser limitante para vídeos muito grandes.

6.5 Tabela de Referência Rápida para Parâmetros

Esta tabela oferece um ponto de partida para diferentes cenários. Os valores exatos podem variar e devem ser ajustados experimentalmente.

| Estruturas de Concreto | 30         | 1 - 20                   | 10 - 20     | 4            | Baixa frequência, maior massa.                                     |
| Materiais Compósitos | 120        | 10 - 100                 | 30 - 50     | 6            | Amortecimento variável, alta frequência.                           |
| Objetos Pequenos / Leves | 120 - 240  | 20 - 200                 | 50 - 100    | 7            | Vibrações muito rápidas, requer alto FPS e ganho.                  |

---

7. INTERPRETAÇÃO DE RESULTADOS 📊

A interpretação dos resultados do EVM para análise de tensões residuais é fundamentalmente qualitativa e relativa. O mapa de calor indica regiões com maior ou menor energia de resposta vibracional dentro da banda de frequência analisada.

7.1 Entendendo o Heatmap

*   Interpretação de Cores:
    *   Cores Frias (Azul, Verde Escuro): Indicam regiões com baixa amplitude de vibração ou baixa energia de resposta modal dentro da banda de frequência filtrada. Isso pode significar maior rigidez, menor excitação ou ausência de anomalias.
    *   Cores Quentes (Amarelo, Laranja, Vermelho): Indicam regiões com alta amplitude de vibração ou alta energia de resposta modal dentro da banda de frequência filtrada. Estas são as áreas de maior interesse.
*   Padrões Espaciais:
    *   Gradientes: Mudanças graduais de cor podem indicar variações contínuas na rigidez ou na distribuição de tensão.
    *   Pontos Quentes (Hotspots): Regiões pequenas e intensamente vermelhas podem sugerir concentrações de tensão, pontos de falha, trincas, delaminações ou outras anomalias estruturais que alteram drasticamente a resposta vibracional local.
    *   Linhas/Contornos: Podem delinear áreas de solda, interfaces de materiais ou regiões de transição de geometria onde as tensões residuais são esperadas.
*   Consistência Temporal: O heatmap é uma média temporal. É importante observar o vídeo processado para entender se os "hotspots" são consistentes ao longo do tempo ou se são eventos transitórios.

7.2 Análise dos Arquivos CSV

Os arquivos CSV fornecem dados quantitativos brutos que complementam a visualização do heatmap.

7.2.1 stats.csv

Este arquivo contém estatísticas resumidas do mapa A_RMS (antes da normalização por percentis).

*   Campos: min_rms, max_rms, mean_rms, std_rms, p5_rms, p95_rms.
*   Explicação:
    *   min_rms, max_rms: Valores mínimo e máximo de RMS encontrados no mapa.
    *   mean_rms, std_rms: Média e desvio padrão dos valores de RMS.
    *   p5_rms, p95_rms: Os valores de RMS correspondentes aos percentis 5 e 95, usados na normalização.
*   Uso: Fornece uma visão geral da distribuição da energia vibracional e ajuda a entender a faixa de valores antes da normalização.

7.2.2 pixels.csv

Este arquivo contém o valor A_RMS (normalizado e clamped entre 0 e 1) para cada pixel do mapa de calor.

*   Formato: row, col, normalized_rms_value
*   Uso: Permite análises quantitativas mais aprofundadas em softwares externos (e.g., MATLAB, Excel, Python com Pandas). Pode ser usado para:
    *   Plotar perfis de tensão relativa ao longo de linhas específicas.
    *   Realizar segmentação de regiões com base em limiares de normalized_rms_value.
    *   Comparar a distribuição de tensão relativa entre diferentes amostras ou condições.

7.3 Correlação com Fenômenos Físicos

O heatmap pode ser correlacionado com:

*   Variações de Rigidez: Regiões com menor rigidez (e.g., devido a danos, fadiga, ou material mais flexível) tendem a vibrar com maior amplitude para uma dada excitação, aparecendo como "hotspots".
*   Concentrações de Tensão: Áreas onde as tensões se concentram (e.g., cantos vivos, furos, soldas) podem ter sua rigidez local alterada, influenciando a resposta vibracional.
*   Condições de Contorno: A forma como uma estrutura é fixada ou suportada afeta diretamente seus modos de vibração. Mudanças nas condições de contorno podem ser visíveis.
*   Defeitos de Material: Trincas, delaminações, porosidade ou inclusões podem alterar a integridade estrutural e, consequentemente, a resposta vibracional local.

7.4 Casos de Falso Positivo

É crucial estar ciente de que nem todo "hotspot" no heatmap indica necessariamente uma tensão residual crítica ou um defeito. Falsos positivos podem ocorrer devido a:

*   Ruído Amplificado: Ruído de câmera ou ambiental pode ser amplificado e aparecer como um hotspot.
*   Reflexos/Sombras: Mudanças na iluminação ou reflexos podem ser interpretados como movimento.
*   Movimento de Câmera: Mesmo com tripé, pequenas vibrações da câmera podem ser amplificadas.
*   Movimento de Fundo: Objetos em movimento no fundo podem gerar artefatos.
*   Vibrações Externas: Vibrações de outras fontes não relacionadas à estrutura em análise.
*   Variações de Superfície: Texturas ou irregularidades na superfície podem interagir com a luz e gerar padrões.

Sempre valide os resultados com inspeção visual, conhecimento da estrutura e, se possível, outras técnicas de END.

---

8. VALIDAÇÃO CIENTÍFICA 🧪

A validação da aplicação é um passo crítico para garantir que ela está funcionando conforme o esperado e que os resultados são confiáveis dentro de suas limitações.

8.1 Protocolo de Validação

Um protocolo de validação sistemático deve incluir:

1.  Vídeos Sintéticos: Gerar vídeos com movimentos e frequências conhecidas para verificar a capacidade do EVM de detectar e amplificar esses movimentos.
2.  Vídeos Reais Controlados: Filmar objetos com vibrações induzidas e conhecidas (e.g., um diapasão, uma viga com excitação forçada).
3.  Comparação com Métodos Tradicionais: Se possível, comparar os resultados qualitativos do heatmap com dados de extensômetros, acelerômetros ou outras técnicas de END.
4.  Análise de Sensibilidade: Estudar como os parâmetros (alpha, f_low, f_high) afetam a saída.
5.  Robustez ao Ruído: Testar a aplicação com vídeos contendo diferentes níveis de ruído.

8.2 Vídeo Sintético

O script generate_synthetic_test_video.py é uma ferramenta essencial para a validação.

   Propósito: Criar um ambiente controlado onde a frequência e a amplitude da vibração são conhecidas* e variam de forma previsível. Isso permite verificar se o EVM está amplificando as frequências corretas e se o mapa de calor reflete a distribuição de amplitude esperada.
*   Processo de Geração: O script cria um vídeo de 10 segundos a 30 FPS. Um padrão de faixas horizontais se move verticalmente com uma frequência de 3 Hz. A amplitude desse movimento varia linearmente de 0 pixels (na borda esquerda do frame) a 5 pixels (na borda direita do frame).
*   Resultados Esperados:
    *   Ao processar samples/synthetic_test_video.mp4 com f_low=0.5 Hz, f_high=3.0 Hz e alpha=10-20, o heatmap deve exibir um gradiente horizontal claro.
    *   A cor deve variar de azul/frio (esquerda, amplitude 0) para vermelho/quente (direita, amplitude 5).
    *   O vídeo processado deve mostrar a amplificação do movimento vertical, mais pronunciada à direita.
*   Métricas de Sucesso:
    *   O heatmap reflete o gradiente de amplitude programado.
    *   A banda de frequência correta é isolada.
    *   A amplificação é visível sem artefatos excessivos.

8.3 Experimentos Sistemáticos

Experimento 1: Efeito do Alpha

*   Procedimento: Use o vídeo sintético. Mantenha f_low=0.5 Hz, f_high=3.0 Hz. Varie Ganho Alpha de 5, 10, 20, 50, 100, 200.
*   Resultados Esperados:
    *   Alpha baixo: Pouca amplificação, heatmap fraco.
    *   Alpha moderado (10-50): Gradiente claro, amplificação visível.
    *   Alpha alto (>100): Artefatos visuais (saturação, ruído, distorção), heatmap pode ficar saturado.
*   Conclusão: Identificar a faixa ideal de alpha para visibilidade sem artefatos.

Experimento 2: Bandas de Frequência

*   Procedimento: Use o vídeo sintético. Mantenha Ganho Alpha=20.
    *   Cenário 1 (Correto): f_low=0.5 Hz, f_high=3.0 Hz.
    *   Cenário 2 (Banda Errada - Baixa): f_low=0.1 Hz, f_high=0.2 Hz.
    *   Cenário 3 (Banda Errada - Alta): f_low=5.0 Hz, f_high=8.0 Hz.
    *   Cenário 4 (Banda Larga): f_low=0.1 Hz, f_high=10.0 Hz.
*   Resultados Esperados:
    *   Cenário 1: Gradiente claro.
    *   Cenário 2 e 3: Heatmap uniforme e escuro (pouca ou nenhuma vibração detectada, pois a frequência de 3 Hz está fora da banda).
    *   Cenário 4: Gradiente visível, mas com mais ruído ou movimentos indesejados amplificados.
*   Conclusão: Demonstrar a importância de selecionar a banda de frequência correta.

Experimento 3: Colormaps

*   Procedimento: Use o vídeo sintético com parâmetros EVM ideais. Varie o Colormap entre inferno, turbo, viridis, jet, gray.
*   Resultados Esperados: O padrão de gradiente deve ser o mesmo, mas a percepção visual das diferenças pode variar. inferno e viridis geralmente oferecem melhor percepção de gradiente.
*   Conclusão: Escolher o colormap mais adequado para a visualização e interpretação.

Experimento 4: ROI (Região de Interesse)

*   Procedimento: Use o vídeo sintético com parâmetros EVM ideais.
    *   Cenário 1 (Sem ROI): Processar o vídeo completo.
    *   Cenário 2 (ROI Esquerda): Definir X_min=0, Y_min=0, X_max=300, Y_max=480.
    *   Cenário 3 (ROI Direita): Definir X_min=340, Y_min=0, X_max=640, Y_max=480.
*   Resultados Esperados:
    *   Cenário 1: Heatmap completo com gradiente.
    *   Cenário 2: Heatmap apenas na região esquerda, mostrando cores frias.
    *   Cenário 3: Heatmap apenas na região direita, mostrando cores quentes.
*   Conclusão: Demonstrar a capacidade de focar a análise em áreas específicas e a redução do tempo de processamento.

8.4 Métricas de Qualidade

A qualidade dos resultados pode ser avaliada por:

*   SNR (Signal-to-Noise Ratio): Visualmente, a clareza do padrão de vibração em relação ao ruído.
*   Fidelidade do Padrão: Quão bem o heatmap reflete o padrão de vibração esperado (em vídeos sintéticos).
*   Ausência de Artefatos: Mínima presença de saturação, distorção ou ruído excessivo.
*   Consistência: Resultados reproduzíveis sob as mesmas condições.

---

9. CASOS DE USO REAIS 🏭

Esta seção explora como o aplicativo pode ser aplicado em cenários de engenharia do mundo real, com sugestões de setup e interpretação.

9.1 Análise de Vigas Metálicas

*   Setup: Filmar uma viga metálica (e.g., aço, alumínio) sob excitação (e.g., impacto leve, vibração de máquina próxima). A câmera deve estar fixa em um tripé, focada na viga.
*   Parâmetros Sugeridos:
    *   FPS: 60-120 (para capturar frequências mais altas)
    *   f_low: 5 Hz, f_high: 50 Hz (faixa comum para modos de viga)
    *   Ganho Alpha: 15-25
*   Interpretação: Hotspots podem indicar:
    *   Regiões de menor rigidez (e.g., devido a corrosão, fadiga).
    *   Pontos de concentração de tensão (e.g., perto de furos, soldas).
    *   Modos de vibração específicos da viga.

9.2 Análise de Soldas

*   Setup: Filmar a região de uma solda em uma estrutura metálica. A excitação pode ser por impacto ou vibração ambiente. Uma ROI pode ser útil para focar na solda.
*   Parâmetros Sugeridos:
    *   FPS: 60
    *   f_low: 1 Hz, f_high: 20 Hz (frequências de interesse para defeitos)
    *   Ganho Alpha: 20-40
*   Interpretação: Variações significativas no heatmap ao longo da linha de solda podem indicar:
    *   Defeitos na solda (porosidade, falta de fusão).
    *   Tensões residuais elevadas na zona afetada pelo calor.
    *   Diferenças de rigidez entre o metal base e o cordão de solda.

9.3 Análise de Compósitos

*   Setup: Filmar uma placa de material compósito (e.g., fibra de carbono, fibra de vidro) sob vibração. Compósitos podem ter modos de vibração complexos e amortecimento.
*   Parâmetros Sugeridos:
    *   FPS: 120-240 (para capturar modos de alta frequência)
    *   f_low: 10 Hz, f_high: 100 Hz
    *   Ganho Alpha: 30-50 (compósitos podem ter menor amplitude de vibração)
*   Interpretação: Hotspots ou padrões anômalos podem sugerir:
    *   Delaminações ou descolamentos de camadas.
    *   Danos por impacto (impact damage).
    *   Variações na distribuição de fibras ou resina.

9.4 Análise de Estruturas de Concreto

*   Setup: Filmar uma seção de uma estrutura de concreto (e.g., pilar, laje) sob vibração ambiente ou induzida.
*   Parâmetros Sugeridos:
    *   FPS: 30-60 (concreto geralmente vibra em baixas frequências)
    *   f_low: 0.5 Hz, f_high: 10 Hz
    *   Ganho Alpha: 10-20
*   Interpretação:
    *   Regiões com maior vibração podem indicar fissuras, desagregação ou áreas com menor integridade estrutural.
    *   Variações de rigidez devido a danos internos.

9.5 Detecção de Trincas

*   Setup: Filmar uma área onde uma trinca é suspeita ou conhecida. A excitação deve ser tal que a trinca possa "abrir e fechar" ou influenciar a vibração local.
*   Parâmetros Sugeridos:
    *   FPS: 60-120
    *   f_low: 5 Hz, f_high: 50 Hz (dependendo do material e tamanho da trinca)
    *   Ganho Alpha: 20-40
*   Interpretação: Uma trinca pode atuar como uma descontinuidade na rigidez, levando a uma concentração de energia vibracional em suas proximidades. Um hotspot ou um padrão de vibração anômalo ao redor da trinca pode ser observado.

---

10. MELHORES PRÁTICAS ✅

Para obter os melhores resultados com o aplicativo EVM, a qualidade do vídeo de entrada e a configuração experimental são cruciais.

10.1 Captura de Vídeo

*   Câmera e Estabilização:
    *   Tripé Robusto: Essencial para eliminar o movimento da câmera. Qualquer movimento, mesmo que mínimo, será amplificado.
    *   Ângulo e Distância: Posicione a câmera perpendicularmente à superfície de interesse, se possível. Mantenha uma distância que permita capturar a área desejada com boa resolução.
    *   Foco Fixo: Desabilite o autofoco da câmera e defina o foco manualmente na superfície da estrutura. Variações de foco são interpretadas como variações de intensidade.
*   Iluminação:
    *   Constante e Uniforme: Use fontes de luz contínuas e estáveis. Evite luzes piscantes (e.g., fluorescentes com flicker) ou sombras em movimento.
    *   Sem Reflexos: Posicione as luzes para evitar reflexos especulares na superfície do objeto, que podem gerar artefatos.
*   Configurações da Câmera:
    *   Resolução: Use a maior resolução possível que seu hardware possa processar (e.g., 1080p).
    *   FPS (Frames por Segundo):
        *   Mínimo: 30 FPS.
        *   Recomendado: 60 FPS ou superior (120, 240 FPS) para capturar vibrações de alta frequência e evitar aliasing. Lembre-se do critério de Nyquist (f_high < FPS/2).
    *   Exposição e ISO: Ajuste manualmente para evitar variações automáticas que podem introduzir ruído. Mantenha o ISO o mais baixo possível para reduzir o ruído do sensor.
    *   Velocidade do Obturador (Shutter Speed): Use uma velocidade de obturador rápida (e.g., 1/250s ou mais rápido) para minimizar o desfoque de movimento (motion blur), especialmente para objetos que vibram rapidamente.
*   Formato de Vídeo e Codec:
    *   Use formatos com baixa compressão (e.g., .MOV, .MP4 com codec H.264 de alta qualidade). Evite codecs com alta compressão que podem introduzir artefatos.
*   Duração do Vídeo:
    *   Recomendação: 10 a 60 segundos são geralmente suficientes. Vídeos muito longos aumentam o tempo de processamento e o uso de memória.
    *   Capture tempo suficiente para observar vários ciclos da vibração de interesse.

10.2 Excitação da Estrutura

*   Excitação Natural: Use vibrações ambientais (e.g., tráfego, vento, máquinas próximas) se a estrutura já estiver vibrando.
*   Excitação Induzida:
    *   Impacto: Um impacto leve (e.g., martelo de borracha) pode excitar os modos naturais da estrutura.
    *   Vibrador Eletrodinâmico: Para excitação controlada em frequências específicas.
    *   Ruído Branco: Excitação aleatória para ativar múltiplos modos.
*   Evite Excitação Excessiva: Não cause danos à estrutura durante a excitação. O EVM funciona melhor com movimentos sutis.

10.3 Processamento

*   Ajuste Fino dos Parâmetros: Comece com valores conservadores para alpha e ajuste as frequências de filtro com base no conhecimento da estrutura ou em uma análise preliminar.
*   ROI: Utilize a Região de Interesse para focar a análise e reduzir o tempo de processamento, especialmente em vídeos grandes.
*   Iteração: O processamento EVM é iterativo. Experimente diferentes parâmetros para encontrar a melhor visualização.

10.4 Análise dos Resultados

*   Contexto: Sempre interprete o heatmap no contexto do conhecimento da estrutura, material e condições de carga.
*   Validação Cruzada: Se possível, compare os hotspots com inspeções visuais, dados de sensores ou modelos de elementos finitos.
*   Disclaimers: Lembre-se sempre da natureza qualitativa dos resultados.

---

11. SOLUÇÃO DE PROBLEMAS COMPLETA 🆘

Esta seção aborda problemas comuns que podem surgir durante a instalação, execução ou interpretação dos resultados do aplicativo EVM.

1.  Problema: FPS inválido detectado (0 ou valor irrealista)
    *   Causa: O codec do vídeo ou o OpenCV não conseguiram ler o FPS corretamente do arquivo.
    *   Solução: Insira o FPS correto manualmente no campo "FPS do Vídeo (Detectado)" na sidebar. Se não souber, 30 FPS é um bom ponto de partida.
    *   Prevenção: Use vídeos com metadados de FPS bem definidos ou codecs padrão.

2.  Problema: Erro: f_high deve ser menor que FPS/2 (Critério de Nyquist)
    *   Causa: A frequência alta do filtro (f_high) é igual ou superior à frequência de Nyquist (FPS/2).
    *   Solução: Reduza o valor de f_high na sidebar para que seja estritamente menor que FPS/2.
    *   Prevenção: Sempre verifique o FPS do seu vídeo e defina f_high de acordo. Para 30 FPS, f_high < 15 Hz. Para 60 FPS, f_high < 30 Hz.

3.  Problema: Processamento muito lento
    *   Causa: Vídeo muito longo, alta resolução, ou hardware limitado.
    *   Solução:
        *   Reduza o "Máximo de frames para preview" na sidebar para testes rápidos.
        *   Use vídeos mais curtos ou de menor resolução.
        *   Defina uma ROI (Região de Interesse) para processar apenas uma parte do frame.
        *   Considere um hardware com mais RAM ou CPU mais potente.
    *   Prevenção: Otimize a duração e resolução do vídeo de captura.

4.  Problema: Mapa de calor uniforme (todo azul/escuro) ou sem gradiente
    *   Causa:
        *   A banda de frequência do filtro (f_low, f_high) não inclui a frequência da vibração de interesse.
        *   A vibração é muito sutil e o Ganho Alpha é muito baixo.
        *   Não há vibração significativa no vídeo.
        *   O vídeo tem muito ruído que está mascarando a vibração.
    *   Solução:
        *   Ajuste f_low e f_high para cobrir a frequência esperada da vibração.
        *   Aumente o Ganho Alpha gradualmente.
        *   Verifique se há vibração real no vídeo.
        *   Melhore a qualidade da captura de vídeo (iluminação, estabilidade).
    *   Prevenção: Conheça as frequências naturais da sua estrutura.

5.  Problema: Artefatos visuais (cores estouradas, distorções, "ondas")
    *   Causa: Ganho Alpha muito alto, ou ruído excessivo no vídeo sendo amplificado.
    *   Solução: Reduza o Ganho Alpha. Melhore a qualidade da captura de vídeo (iluminação, estabilidade da câmera, foco).
    *   Prevenção: Comece com alpha baixo e aumente gradualmente.

6.  Problema: Erro de memória (MemoryError)
    *   Causa: O vídeo é muito grande (muitos frames ou alta resolução) e excede a RAM disponível.
    *   Solução:
        *   Reduza a resolução do vídeo de entrada.
        *   Use vídeos mais curtos.
        *   Aumente a RAM do seu sistema.
        *   Defina uma ROI para reduzir o tamanho dos dados processados.
    *   Prevenção: Monitore o uso de RAM para vídeos grandes.

7.  Problema: Codec de vídeo não suportado
    *   Causa: O OpenCV não consegue decodificar o formato ou codec do vídeo.
    *   Solução: Converta o vídeo para um formato mais comum como MP4 (H.264) usando ferramentas como FFmpeg ou conversores online.
    *   Prevenção: Capture vídeos em formatos amplamente suportados.

8.  Problema: Upload de vídeo falha ou demora muito
    *   Causa: Arquivo de vídeo muito grande, conexão de rede lenta (se Streamlit estiver em servidor remoto).
    *   Solução: Reduza o tamanho do arquivo de vídeo. Execute o Streamlit localmente.
    *   Prevenção: Otimize o tamanho do vídeo antes do upload.

9.  Problema: Resultados inconsistentes entre execuções
    *   Causa: Variações na captura de vídeo (iluminação, movimento), ou parâmetros de filtro ligeiramente diferentes.
    *   Solução: Garanta condições de captura idênticas. Use os mesmos parâmetros de processamento.
    *   Prevenção: Padronize o processo de captura e os parâmetros de análise.

10. Problema: Overlay do heatmap não visível ou muito fraco
    *   Causa: Opacidade do Overlay muito baixa, ou heatmap muito escuro devido a Ganho Alpha baixo ou normalização inadequada.
    *   Solução: Aumente a Opacidade do Overlay. Ajuste Ganho Alpha e os percentis de normalização.
    *   Prevenção: Experimente diferentes valores de opacidade e colormaps.

11. Problema: Filtro temporal instável ou com comportamento inesperado
    *   Causa: Ordem do filtro muito alta, ou f_low e f_high muito próximos, criando uma banda de passagem muito estreita.
    *   Solução: Reduza a Ordem do filtro. Aumente ligeiramente a largura da banda de frequência.
    *   Prevenção: Use ordens de filtro moderadas (4-6).

12. Problema: Cores do heatmap saturadas (muito vermelho/azul)
    *   Causa: Normalização inadequada (percentis muito próximos ou muito distantes), ou Ganho Alpha muito alto.
    *   Solução: Ajuste os percentis de normalização (p5 e p95). Reduza Ganho Alpha.
    *   Prevenção: Use os percentis padrão (5 e 95) como ponto de partida.

13. Problema: Ruído excessivo no heatmap
    *   Causa: Vídeo de baixa qualidade, alto ISO na câmera, iluminação inconsistente, ou Ganho Alpha amplificando o ruído.
    *   Solução: Melhore a qualidade da captura de vídeo. Reduza Ganho Alpha. Considere pré-processamento de vídeo para redução de ruído.
    *   Prevenção: Siga as melhores práticas de captura de vídeo.

14. Problema: Gradiente de cores invertido no heatmap
    *   Causa: Colormap selecionado pode ter uma ordem de cores que não corresponde à expectativa (e.g., gray_r em vez de gray).
    *   Solução: Experimente outros colormaps ou verifique se o colormap tem uma versão reversa (_r).
    *   Prevenção: Escolha colormaps perceptualmente uniformes como inferno ou viridis.

15. Problema: Estatísticas CSV incorretas ou inesperadas
    *   Causa: Erro no cálculo, ou interpretação errada dos dados.
    *   Solução: Verifique a lógica de cálculo no código. Entenda o que cada estatística representa.
    *   Prevenção: Revise a seção 7.2.1.

16. Problema: Arquivo CSV de pixels vazio ou com poucos dados
    *   Causa: O processamento falhou antes da exportação, ou uma ROI muito pequena foi definida.
    *   Solução: Verifique se o processamento foi concluído com sucesso. Ajuste a ROI.
    *   Prevenção: Monitore a barra de progresso.

17. Problema: Vídeo de saída corrompido ou não reproduz
    *   Causa: Problema com o codec de escrita do OpenCV, ou arquivo incompleto.
    *   Solução: Tente um codec diferente (e.g., XVID em vez de mp4v no código, se estiver editando). Verifique se o arquivo foi salvo completamente.
    *   Prevenção: Use codecs amplamente suportados.

18. Problema: Aplicação Streamlit trava ou fecha inesperadamente
    *   Causa: Erro de código, erro de memória, ou dependências conflitantes.
    *   Solução: Verifique o console onde o Streamlit foi iniciado para mensagens de erro. Reinicie o ambiente virtual.
    *   Prevenção: Mantenha as dependências atualizadas e use um ambiente virtual.

19. Problema: Dependências faltando após instalação
    *   Causa: pip install -r requirements.txt não foi executado, ou o ambiente virtual não está ativado.
    *   Solução: Ative o ambiente virtual e execute pip install -r requirements.txt novamente.
    *   Prevenção: Siga rigorosamente os passos de instalação.

20. Problema: Porta 8501 já em uso
    *   Causa: Outra instância do Streamlit ou outro serviço está usando a porta padrão.
    *   Solução: Inicie o Streamlit em uma porta diferente: streamlit run streamlit_app.py --server.port 8502.
    *   Prevenção: Verifique se não há outras aplicações Streamlit rodando.

---

12. ESTRUTURA DO CÓDIGO 💻

O projeto é organizado em arquivos Python que implementam as diferentes funcionalidades, com o Streamlit orquestrando a interface e o fluxo de trabalho.

12.1 Organização dos Arquivos

`
evm-stress-analysis/
├── README.md                         # Documentação completa do projeto
├── streamlit_app.py                  # Script principal da aplicação Streamlit
├── generate_synthetic_test_video.py  # Script para gerar vídeo de teste sintético
├── requirements.txt                  # Lista de dependências Python
├── samples/                          # Pasta para vídeos de exemplo (criada automaticamente)
│   └── synthetic_test_video.mp4      # Vídeo de teste sintético
└── outputs/                          # Pasta para resultados gerados (criada automaticamente)
    ├── processed_video.mp4           # Vídeo com overlay do heatmap
    ├── heatmap.png                   # Imagem estática do heatmap
    ├── stats.csv                     # Estatísticas do mapa RMS
    └── pixels.csv                    # Valores RMS por pixel
`

12.2 Módulos Principais

streamlit_app.py

Este é o coração da aplicação, contendo a lógica da interface e a integração com as funções de processamento.

*   load_video(uploaded_file, max_frames_to_process):
    *   Propósito: Carrega o vídeo do uploaded_file, extrai frames, converte para escala de cinza e normaliza. Retorna o stack de frames, FPS e dimensões.
    *   Fluxo: Usa cv2.VideoCapture para ler o vídeo.
*   apply_temporal_bandpass(frames_stack, fps, f_low, f_high, order):
    *   Propósito: Aplica o filtro passa-banda Butterworth a cada pixel ao longo do tempo.
    *   Fluxo: Utiliza scipy.signal.butter para projetar o filtro e scipy.signal.sosfiltfilt para aplicar o filtro de forma bidirecional (para evitar atraso de fase).
*   compute_rms_map(filtered_signal):
    *   Propósito: Calcula o valor RMS do sinal filtrado para cada pixel.
    *   Fluxo: Usa numpy.mean e numpy.sqrt no eixo temporal.
*   normalize_map(rms_map, p_low, p_high):
    *   Propósito: Normaliza o mapa RMS usando percentis para robustez.
    *   Fluxo: Calcula os percentis p_low e p_high usando numpy.percentile e aplica a fórmula de normalização, clampeando os valores entre 0 e 1.
*   generate_heatmap_overlay(original_frame, normalized_rms_map, colormap_name, opacity):
    *   Propósito: Cria o heatmap a partir do mapa RMS normalizado e o sobrepõe a um frame original.
    *   Fluxo: Usa matplotlib.colormaps para aplicar o colormap e cv2.addWeighted para a sobreposição com opacidade.
*   write_output_video(output_path, frames_with_overlay, fps, dimensions):
    *   Propósito: Grava a sequência de frames com overlay em um novo arquivo de vídeo.
    *   Fluxo: Utiliza cv2.VideoWriter com o codec mp4v.
*   export_data(rms_map_normalized, stats_df):
    *   Propósito: Exporta o heatmap estático em PNG e os dados RMS/estatísticas em CSV.
    *   Fluxo: Usa matplotlib.pyplot.imsave e pandas.DataFrame.to_csv.

generate_synthetic_test_video.py

*   Propósito: Script autônomo para criar um vídeo de teste com vibração controlada para validação.
*   Como funciona: Gera frames com um padrão de faixas e aplica um deslocamento vertical senoidal cuja amplitude varia espacialmente. Usa cv2.VideoWriter para salvar o vídeo.

12.3 Fluxo de Dados Detalhado (Textual)

1.  streamlit_app.py:
    *   st.file_uploader recebe uploaded_file.
    *   load_video processa uploaded_file -> frames_stack, fps, dimensions.
    *   apply_temporal_bandpass processa frames_stack, fps, f_low, f_high, order -> filtered_signal.
    *   compute_rms_map processa filtered_signal -> rms_map.
    *   normalize_map processa rms_map, p_low, p_high -> rms_map_normalized.
    *   Loop sobre frames_stack e rms_map_normalized:
        *   generate_heatmap_overlay processa original_frame, rms_map_normalized, colormap_name, opacity -> frame_with_overlay.
        *   frame_with_overlay é adicionado a uma lista.
    *   write_output_video processa a lista de frames_with_overlay, fps, dimensions -> processed_video.mp4.
    *   export_data processa rms_map_normalized, stats_df -> heatmap.png, stats.csv, pixels.csv.
    *   Streamlit exibe resultados e botões de download.

12.4 Pontos de Extensão

O código foi projetado para ser modular, permitindo futuras extensões:

*   Implementação de Pirâmide Laplaciana: A função apply_temporal_bandpass poderia ser modificada para operar em múltiplos níveis de uma pirâmide Laplaciana, melhorando a SNR.
*   EVM Baseado em Fase: Uma abordagem mais avançada (Wadhwa et al., 2014) que amplifica a fase do sinal, mais robusta a variações de iluminação.
*   Correção de Movimento: Adicionar um módulo de estabilização de vídeo antes do EVM para lidar com pequenos movimentos de câmera.
*   Processamento em Blocos: Para vídeos muito grandes, o load_video e o pipeline de processamento poderiam ser adaptados para carregar e processar frames em blocos temporais.
*   Novas Métricas: Adicionar outras métricas de análise vibracional (e.g., STFT por pixel, análise de coerência).
*   ROI Dinâmica: Permitir que o usuário desenhe a ROI diretamente na pré-visualização do frame.

---

13. DESENVOLVIMENTO FUTURO 🚀

Este projeto é uma base sólida para futuras melhorias e expansões. Abaixo estão algumas das melhorias planejadas e ideias para o roadmap.

13.1 Melhorias Planejadas

1.  Implementação de Pirâmide Laplaciana: Adicionar a decomposição espacial multi-escala para melhoria da relação sinal-ruído (SNR) e amplificação mais robusta.
2.  EVM Baseado em Fase (Phase-based EVM): Implementar a técnica de Wadhwa et al. (2014), que é mais robusta a ruído e variações de iluminação.
3.  Análise Tempo-Frequência Localizada (STFT por Região): Permitir que o usuário selecione uma região e visualize o espectrograma (STFT) para identificar frequências dominantes localmente.
4.  Correção de Movimento (Estabilização de Vídeo): Integrar algoritmos de estabilização de vídeo (e.g., usando cv2.Tracker ou cv2.estimateAffine2D) como um pré-processamento opcional.
5.  Processamento em Blocos (Chunking): Otimizar o uso de memória para vídeos muito longos ou de alta resolução, processando-os em blocos temporais.
6.  Aceleração por GPU: Explorar o uso de bibliotecas como cupy ou PyTorch para acelerar o processamento em GPUs.
7.  Comparação Multi-Vídeo: Funcionalidade para carregar e comparar heatmaps de múltiplos vídeos (e.g., antes e depois de um reparo).
8.  Ajuste Automático de Parâmetros: Desenvolver algoritmos para sugerir parâmetros EVM (frequências, alpha) com base na análise espectral preliminar do vídeo.
9.  Visualização 3D de Deslocamentos: Para casos específicos, tentar reconstruir um campo de deslocamentos 3D a partir de múltiplas câmeras ou modelos.
10. Integração com FEA (Análise de Elementos Finitos): Possibilidade de importar resultados de simulações FEA para comparação direta com os mapas de vibração.
11. Melhorias na Interface do Usuário:
    *   Desenho de ROI interativo na pré-visualização.
    *   Gráficos interativos para análise de sinal temporal de um pixel selecionado.
    *   Opções de filtro mais avançadas (e.g., Chebyshev, Elliptic).
12. Machine Learning para Detecção de Anomalias: Treinar modelos de ML para identificar padrões de heatmap associados a defeitos conhecidos.
13. Versão Mobile App: Explorar a possibilidade de uma versão simplificada para dispositivos móveis.

13.2 Como Contribuir

Contribuições são bem-vindas! Se você tiver ideias para melhorias, detecção de bugs ou quiser implementar novas funcionalidades, siga estas diretrizes:

1.  Fork o repositório.
2.  Crie uma branch para sua feature (git checkout -b feature/MinhaNovaFeature).
3.  Implemente suas mudanças e teste-as cuidadosamente.
4.  Commit suas mudanças (git commit -m 'feat: Adiciona nova funcionalidade X').
5.  Push para a branch (git push origin feature/MinhaNovaFeature).
6.  Abra um Pull Request descrevendo suas mudanças.

13.3 Roadmap

*   V1.0 (Atual): Implementação básica do EVM (filtragem temporal direta), cálculo RMS, normalização, heatmap e interface Streamlit.
*   V1.1 (Curto Prazo):
    *   Implementação de Pirâmide Laplaciana para decomposição espacial.
    *   Melhorias na gestão de memória para vídeos maiores.
    *   ROI interativa na UI.
*   V1.2 (Médio Prazo):
    *   EVM Baseado em Fase.
    *   Módulo de estabilização de vídeo.
    *   Análise STFT por região.
*   V2.0 (Longo Prazo):
    *   Aceleração por GPU.
    *   Integração com Machine Learning para detecção de anomalias.
    *   Possível integração com modelos FEA.

---

14. REFERÊNCIAS BIBLIOGRÁFICAS COMPLETAS 📖

Esta seção lista as principais referências científicas e técnicas que fundamentam este projeto.

14.1 Papers Fundamentais

1.  Wu, H.-Y., Rubinstein, M., Shih, E., Guttag, J., Durand, F., & Freeman, W. T. (2012). Eulerian Video Magnification for Revealing Subtle Changes in the World. ACM Transactions on Graphics (TOG), 31(4), 1-8. DOI: 10.1145/2185520.2185561
    *   O paper original que introduziu o conceito de Eulerian Video Magnification.

2.  Wadhwa, N., Rubinstein, M., Durand, F., & Freeman, W. T. (2014). Phase-Based Video Motion Processing. ACM Transactions on Graphics (TOG), 32(4), 1-10. DOI: 10.1145/2461912.2461966
    *   Introduz uma abordagem baseada em fase para EVM, que é mais robusta a variações de iluminação e ruído.

3.  Davis, J., & Bobick, A. F. (1997). The representation and recognition of human action using temporal templates. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition. DOI: 10.1109/CVPR.1997.609314
    *   Trabalho anterior sobre análise de movimento em vídeo que influenciou o desenvolvimento de técnicas como o EVM.

4.  Fleet, D. J., & Jepson, A. D. (1990). Computation of component image velocity from local phase information. International Journal of Computer Vision, 5(1), 77-104. DOI: 10.1007/BF00127814
    *   Fundamentos da análise de movimento baseada em fase.

5.  Adelson, E. H., & Bergen, J. R. (1985). Spatiotemporal energy models for the perception of motion. Journal of the Optical Society of America A, 2(2), 284-299. DOI: 10.1364/JOSAA.2.000284
    *   Conceitos de filtros espaço-temporais que são a base para o EVM.

6.  Simon, D. (1994). Modal Analysis of Structures. Butterworth-Heinemann.
    *   Livro clássico sobre análise modal, relevante para a conexão entre vibração e propriedades estruturais.

14.2 Livros Técnicos

1.  Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal Processing. Pearson Education.
    *   Referência fundamental para processamento digital de sinais, incluindo filtros e análise de frequência.

2.  Gere, J. M., & Goodno, B. J. (2012). Mechanics of Materials. Cengage Learning.
    *   Livro texto sobre mecânica dos materiais, essencial para entender tensões e deformações.

3.  Jain, R. C., Kasturi, R., & Schunck, B. G. (1995). Machine Vision. McGraw-Hill.
    *   Abrange conceitos de visão computacional, incluindo processamento de imagem e análise de movimento.

4.  Inman, D. J. (2017). Engineering Vibration. Pearson.
    *   Livro abrangente sobre vibrações mecânicas, modos naturais e resposta dinâmica de estruturas.

5.  Nondestructive Testing Handbook (Vol. 1-10). American Society for Nondestructive Testing (ASNT).
    *   Série de livros sobre diversas técnicas de END, fornecendo contexto para a aplicação do EVM.

14.3 Recursos Online

*   MIT EVM Project Page: http://people.csail.mit.edu/mrub/vidmag/
    *   Página oficial do projeto EVM do MIT com vídeos de demonstração e informações adicionais.
*   OpenCV Documentation: https://docs.opencv.org/
    *   Documentação da biblioteca OpenCV para visão computacional.
*   SciPy Documentation: https://docs.scipy.org/
    *   Documentação da biblioteca SciPy para computação científica, incluindo processamento de sinais.
*   Streamlit Documentation: https://docs.streamlit.io/
    *   Documentação oficial do framework Streamlit.
*   Matplotlib Documentation: https://matplotlib.org/stable/contents.html
    *   Documentação da biblioteca Matplotlib para plotagem e visualização de dados.

---

15. APÊNDICES 📚

Apêndice A: Glossário de Termos

*   Aliasing: Fenômeno que ocorre quando um sinal é amostrado a uma taxa inferior à frequência de Nyquist, resultando em uma representação distorcida da frequência original.
*   Alpha (α): Fator de ganho usado no EVM para amplificar as variações temporais filtradas.
*   Análise Modal: Estudo das características dinâmicas de uma estrutura (frequências naturais, modos de vibração, amortecimento).
*   A_RMS (Root Mean Square): Valor quadrático médio; uma medida da magnitude média de um sinal variável no tempo. Usado aqui para quantificar a energia vibracional por pixel.
*   Banda Passante (Band-Pass): Faixa de frequências que um filtro permite passar, atenuando as frequências fora dessa faixa.
*   Butterworth Filter: Tipo de filtro eletrônico ou digital conhecido por ter uma resposta de frequência o mais plana possível na banda passante.
*   Colormap: Esquema de cores usado para mapear valores numéricos para cores em uma visualização (e.g., heatmap).
*   Compressão de Vídeo: Redução do tamanho de um arquivo de vídeo, que pode introduzir artefatos e reduzir a qualidade.
*   Critério de Nyquist: Princípio que afirma que a frequência de amostragem deve ser pelo menos o dobro da frequência mais alta presente no sinal para evitar aliasing.
*   Decomposição Espacial: Processo de separar uma imagem ou vídeo em diferentes componentes baseados em suas frequências espaciais (detalhes finos vs. grosseiros).
*   Delaminação: Separação de camadas em materiais compósitos, um tipo de defeito.
*   Deslocamento: Mudança na posição de um ponto ou objeto.
*   Eulerian Approach: Método de análise que observa as mudanças em pontos fixos no espaço ao longo do tempo.
*   Eulerian Video Magnification (EVM): Técnica para amplificar variações temporais sutis em vídeos observando pontos fixos no espaço.
*   Excitação: Aplicação de uma força ou movimento a uma estrutura para induzir vibração.
*   Filtro Temporal: Processo que modifica as componentes de frequência de um sinal ao longo do tempo.
*   FPS (Frames por Segundo): Taxa na qual os quadros de um vídeo são exibidos ou capturados.
*   Frequência Natural: Frequência na qual um sistema tende a vibrar quando perturbado e deixado livre para oscilar.
*   Heatmap: Representação gráfica de dados onde os valores individuais em uma matriz são representados como cores.
*   ISO: Sensibilidade do sensor da câmera à luz. ISO alto aumenta o ruído.
*   Lagrangian Approach: Método de análise que rastreia o movimento de objetos ou pontos específicos no espaço ao longo do tempo.
*   Laplacian Pyramid: Estrutura de imagem multi-escala usada para decomposição espacial, onde cada nível contém os detalhes de uma banda de frequência espacial.
*   Luminância: Componente de brilho de uma cor, frequentemente usada em EVM para simplificar o processamento.
*   Modos de Vibração: Padrões de deformação que uma estrutura assume quando vibra em suas frequências naturais.
*   Motion Blur: Desfoque de movimento; ocorre quando um objeto se move durante o tempo de exposição da câmera.
*   Normalização: Processo de escalar valores para uma faixa padrão (e.g., 0 a 1).
*   Ordem do Filtro: Parâmetro que define a complexidade e a seletividade de um filtro.
*   Percentil: Medida estatística que indica o valor abaixo do qual uma dada porcentagem de observações em um grupo de observações cai.
*   Phase-Based EVM: Variação do EVM que amplifica as variações de fase do sinal de vídeo, geralmente mais robusta.
*   Pirâmide Gaussiana: Estrutura de imagem multi-escala criada por sucessivas aplicações de filtros Gaussianos e subamostragem.
*   ROI (Região de Interesse): Uma área específica dentro de uma imagem ou vídeo selecionada para análise.
*   Ruído: Informação indesejada que interfere na clareza de um sinal.
*   Saturação: Condição onde os valores de pixel atingem o limite máximo (e.g., 255 para 8-bit), resultando em perda de detalhes.
*   SNR (Signal-to-Noise Ratio): Relação entre a potência do sinal desejado e a potência do ruído.
*   sosfiltfilt: Função da biblioteca SciPy para aplicar um filtro digital de forma bidirecional, eliminando o atraso de fase.
*   STFT (Short-Time Fourier Transform): Análise de Fourier aplicada a segmentos curtos de um sinal para analisar como suas frequências mudam ao longo do tempo.
*   Streamlit: Framework Python de código aberto para criar aplicativos web interativos para ciência de dados e machine learning.
*   Tensões Residuais: Tensões que permanecem em um material ou estrutura na ausência de cargas externas.
*   Tripé: Suporte de três pernas para estabilizar uma câmera.

Apêndice B: FAQ Expandido

1.  O que este aplicativo realmente mede?
       Ele mede a intensidade relativa da resposta vibracional (energia RMS) de cada pixel do vídeo dentro de uma banda de frequência específica. Isso é um índice* que pode ser correlacionado com variações de rigidez ou tensões.

2.  Posso obter valores de tensão em MPa?
       Não diretamente. O aplicativo fornece um índice qualitativo/relativo*. Para valores em MPa, você precisaria de calibração com extensômetros, modelos de elementos finitos e conhecimento das propriedades do material.

3.  Por que meu heatmap está todo azul/escuro?
    *   Provavelmente, a banda de frequência do filtro (f_low, f_high) não está capturando a vibração de interesse, ou o Ganho Alpha é muito baixo. Ajuste esses parâmetros.

4.  Por que meu vídeo processado tem artefatos estranhos?
    *   O Ganho Alpha pode estar muito alto, amplificando ruído ou movimentos indesejados. Reduza-o. Também pode ser devido a iluminação inconsistente ou movimento da câmera.

5.  Qual FPS devo usar para gravar meu vídeo?
    *   Mínimo de 30 FPS. Para vibrações mais rápidas, 60 FPS, 120 FPS ou até mais são recomendados. Lembre-se que f_high deve ser < FPS/2.

6.  Preciso de um tripé?
    *   Sim, absolutamente. Qualquer movimento da câmera será amplificado e pode mascarar as vibrações reais.

7.  A iluminação é importante?
    *   Sim, muito. Variações de iluminação são interpretadas como variações de intensidade de pixel e serão amplificadas. Use iluminação constante e uniforme.

8.  Posso usar vídeos do meu celular?
    *   Sim, desde que a qualidade seja boa (alta resolução, bom FPS, sem compressão excessiva) e o celular esteja totalmente estabilizado (em um tripé).

9.  O que é o "Ganho Alpha"?
    *   É o fator pelo qual as variações temporais filtradas são multiplicadas. Um alpha de 20 significa que as variações são amplificadas 20 vezes.

10. O que são "percentis de normalização"?
    *   Eles ajudam a mapear a faixa de valores de RMS para o colormap de forma robusta. p5 e p95 ignoram os 5% menores e 5% maiores valores, respectivamente, para evitar que outliers saturem o mapa de calor.

11. Posso analisar apenas uma parte do vídeo?
    *   Sim, use os campos de ROI (Região de Interesse) na sidebar para definir as coordenadas X_min, Y_min, X_max, Y_max.

12. O que fazer se o aplicativo estiver muito lento?
    *   Reduza o "Máximo de frames para preview" na sidebar. Use vídeos mais curtos ou de menor resolução. Considere um hardware mais potente.

13. Por que o vídeo sintético é importante?
       Ele permite validar a aplicação em um cenário controlado, onde a frequência e a distribuição da amplitude de vibração são conhecidas*.

14. Qual colormap devo usar?
    *   inferno, viridis e plasma são geralmente recomendados para visualização científica por serem perceptualmente uniformes. turbo oferece alto contraste.

15. O que significa "Ordem do filtro"?
    *   Controla a "nitidez" do filtro. Ordens mais altas têm transições mais abruptas entre as bandas, mas podem introduzir oscilações.

16. Posso usar este aplicativo para detectar trincas?
       Ele pode indicar regiões com padrões vibracionais anômalos que podem* estar associados a trincas. No entanto, não é uma ferramenta de detecção de trincas certificada e requer validação por outras técnicas.

17. O que é o critério de Nyquist?
    *   É uma regra fundamental no processamento de sinais que diz que a frequência de amostragem (FPS) deve ser pelo menos o dobro da frequência mais alta que você deseja capturar. Se não for, ocorre aliasing.

18. Por que o opencv-python-headless é usado em vez de opencv-python?
    *   A versão headless não inclui as dependências de GUI (interface gráfica) do OpenCV, tornando-a mais leve e adequada para ambientes de servidor ou onde a GUI do OpenCV não é necessária (como em aplicações Streamlit).

19. Como posso contribuir para o projeto?
    *   Você pode forkar o repositório, implementar melhorias e abrir um Pull Request. Veja a seção 13.2.

20. Este aplicativo é seguro para uso em engenharia crítica?
       NÃO. Este aplicativo é uma ferramenta de pesquisa e análise qualitativa*. Não deve ser usado para tomar decisões em aplicações críticas (aeroespacial, nuclear, médica, etc.) sem validação rigorosa e certificação por métodos aprovados.

Apêndice C: Tabelas de Referência

Comparação de Colormaps (Percepção)

| viridis  | Perceptualmente uniforme, bom para dados científicos, acessível para daltônicos. | Visualização geral de dados, mapas de calor.                                  |
| jet      | Clássico, mas não perceptualmente uniforme, pode criar artefatos visuais.         | Uso histórico, mas geralmente desaconselhado para dados quantitativos.       |
| gray     | Escala de cinza, bom para detalhes finos e para impressão.                        | Análise de detalhes, quando a cor pode distrair.                              |

Propriedades Típicas de Materiais (Vibração)

| Alumínio       | 2700              | 70                    | 20 - 2000                |
| Fibra de Carbono | 1600              | 150 - 250             | 50 - 5000                |

Especificações de Filtros Butterworth (Exemplo)

| 2     | 12 dB                | Bom compromisso, transição razoável. |
| 6     | 36 dB                | Muito seletivo, transição abrupta, pode introduzir mais oscilações na resposta (ringing) se mal projetado. |

Apêndice D: Comandos Úteis

| Comando                                          | Descrição 
| venv\Scripts\activate                          | Ativa o ambiente virtual (Windows).                                    |
| pip list                                       | Lista todas as bibliotecas instaladas no ambiente atual.               |
| streamlit run streamlit_app.py                 | Inicia a aplicação Streamlit.                                          |
| deactivate                                     | Desativa o ambiente virtual.                                           |
| rd /s /q venv                                  | Remove o ambiente virtual (Windows).                                   |

Apêndice E: Troubleshooting Checklist (Fluxograma Textual)

`
INÍCIO
  |
  V
[Problema: Aplicação não inicia ou falha no upload?]
  |
  +--- SIM --> [Verificar: Ambiente virtual ativado? Dependências instaladas? Porta 8501 livre?]
  
  
  |               |               V
  |               V             FIM
  |             FIM
  V
[Problema: Processamento lento ou erro de memória?]
  |
  +--- SIM --> [Verificar: "Máximo de frames para preview" reduzido? Resolução do vídeo? ROI definida?]
  
  |               V
  |             FIM
  V
[Problema: Heatmap uniforme/escuro ou artefatos visuais?]
  |
  +--- SIM --> [Verificar: FPS correto? f_low/f_high corretos (Nyquist)? Ganho Alpha adequado? Iluminação constante? Câmera estável?]
  
  |               V
  |             FIM
  V
[Problema: Resultados CSV incorretos ou vazios?]
  |
  +--- SIM --> [Verificar: Processamento concluído? ROI definida? Lógica de cálculo?]
  
  |               V
  |             FIM
  V
[Problema: Vídeo de saída corrompido?]
  |
  +--- SIM --> [Consultar: Seção 11 - Problema 17]
  |               V
  |             FIM
  V
FIM
`

---

16. LICENÇA E AVISOS LEGAIS ⚖️

16.1 Licença

Este projeto está licenciado sob a Licença MIT. Você é livre para usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do software, desde que inclua a notificação de direitos autorais e esta permissão em todas as cópias ou partes substanciais do software.

`
MIT License

Copyright (c) [Ano] [Seu Nome/Nome da Organização]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`

16.2 Disclaimer Legal

> ⚠️ AVISO LEGAL CRÍTICO:
> ESTE SOFTWARE É FORNECIDO "COMO ESTÁ", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO, MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM FIM ESPECÍFICO E NÃO INFRAÇÃO. EM NENHUM CASO OS AUTORES OU DETENTORES DOS DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM UMA AÇÃO DE CONTRATO, ATO ILÍCITO OU DE OUTRA FORMA, DECORRENTE DE, OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.
>
> Este aplicativo foi desenvolvido para fins educacionais e de pesquisa qualitativa. Os resultados gerados são índices relativos de resposta vibracional e NÃO devem ser interpretados como medições quantitativas de tensão (MPa, Pa). A utilização deste software para tomar decisões críticas de engenharia, segurança ou integridade estrutural é de total responsabilidade do usuário e requer validação por profissionais qualificados e métodos de ensaio certificados.

16.3 Uso Ético

*   Não use este software para enganar ou deturpar dados.
*   Sempre divulgue as limitações da técnica EVM e a natureza qualitativa dos resultados ao apresentar ou publicar análises.
*   Respeite a privacidade ao capturar vídeos, especialmente em ambientes públicos.

16.4 Citação

Se você usar este trabalho em sua pesquisa ou projeto, por favor, cite-o da seguinte forma:

`
[Marcio Fernandes Maciel]. (2026). Aplicativo EVM para Análise de Tensões Residuais. [Link para o Repositório GitHub, se aplicável].
`

---

<footer>
Versão do Projeto: 1.0.0  
Última Atualização: 2026-01-10  
Contato: marciofmaciel@gmail.com  
Agradecimentos: Aos criadores do Streamlit, NumPy, SciPy, OpenCV e Matplotlib por suas excelentes bibliotecas de código aberto.
</footer>
`