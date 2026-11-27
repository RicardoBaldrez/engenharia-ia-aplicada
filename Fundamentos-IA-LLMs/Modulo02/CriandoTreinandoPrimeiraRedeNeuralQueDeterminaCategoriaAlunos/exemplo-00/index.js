/**
 * Classificador de categoria de usuário com rede neural (TensorFlow.js)
 *
 * O que faz:
 *   Prevê em qual categoria um usuário se encaixa: premium, medium ou basic.
 *   A previsão usa idade, cor favorita e localização.
 *
 * Como funciona:
 *   1. Os dados das pessoas são convertidos em números
 *      (idade normalizada + cor e localização em one-hot encoding).
 *   2. Uma rede neural sequencial é treinada com esses exemplos:
 *      7 entradas -> camada oculta (80 neurônios, ReLU) -> saída (3 neurônios, softmax).
 *   3. Para uma nova pessoa, o modelo retorna a probabilidade de cada categoria.
 *
 * Conceitos usados:
 *   tf.sequential, camadas dense, ReLU, softmax, Adam,
 *   categoricalCrossentropy, accuracy, epochs, normalização, one-hot encoding.
 *
 * Como rodar:
 *   npm start
 */

import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  // Cria um modelo sequencial: uma rede neural onde as camadas ficam
  // empilhadas em ordem, e a saída de uma vira a entrada da próxima.
  const model = tf.sequential();

 // Primeira camada -> camada oculta
  // Entrada de 7 posições (idade normalizada + 3 cores + 3 localidades)

  // 80 neurônios = quanto mais neurônios, mais padrões complexos a rede consegue aprender,
  // mas também mais processamento ela usa.
  // Atenção: como a base de treino é pequena, muitos neurônios podem fazer o modelo
  // "decorar" os exemplos (overfitting) em vez de aprender o padrão geral.

  // A ReLU age como um filtro:
  // É como se ela deixasse somente os sinais relevantes seguirem viagem na rede.
  // Se o valor que chegou nesse neurônio é positivo, passa para frente;
  // se for 0 ou negativo, vira 0 (o neurônio "desliga" para essa entrada).
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  // Saída: 3 neurônios -> camada de saída
  // Um para cada categoria [premium, medium, basic]
  // activation: softmax -> normaliza a saída em probabilidades
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // Compilando o modelo -> define como o modelo vai aprender
  // Optimizer Adam (Adaptive Moment Estimation)
  // É um treinador pessoal moderno para redes neurais:
  // ajusta os pesos de forma eficiente e inteligente,
  // aprendendo com o histórico de erros e acertos

  // loss: categoricalCrossentropy
  // Ele compara o que o modelo "acha" (os scores de cada categoria)
  // com a resposta certa
  // A categoria premium será sempre [1, 0, 0]
  // Quanto mais distante a previsão do modelo estiver da resposta correta,
  // maior o erro (loss)
  // categoricalCrossentropy -> exemplos clássicos de uso:
  // classificação de imagens, categorização de usuário, recomendação

  // metrics: valores mostrados durante o treino só para acompanhar o desempenho
  // (não afetam o aprendizado). "accuracy" = % de previsões corretas.
  model.compile({
    optimizer: "Adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Treinamento do modelo -> Treina e faz previsões

  // verbose -> Não mostra o log automático do TensorFlow (0 = nada, 1 = progresso)
  // epochs -> Quantas vezes o modelo vai passar por todos os dados de treino
  // shuffle -> Embaralha os dados a cada época, para o modelo aprender o padrão e não a ordem (e não se viciar)
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      // Roda no fim de cada época e mostra o loss (o tamanho do erro).
      // Se o treino estiver indo bem, o loss vai diminuindo.
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch: ${epoch}: loss = ${log.loss}`),
    },
  });

  return model;
}

async function predict(model, pessoa) {
  // Transforma o array js para o tensor (tfjs)
  const tfinput = tf.tensor2d(pessoa)

  // Faz a predição (output será um vetor de 3 probabilidades)
  const pred = model.predict(tfinput)
  const predArray = await pred.array()
  return predArray[0].map((probabilidade, index) => ({ probabilidade, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, já que a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// Quanto mais dados, melhor!!!
// Assim o algoritmo consegue entender melhor os padrões complexos dos dados
const model = await trainModel(inputXs, outputYs);

// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_mas = 40, então (28 - 25) / (40 -25) = 0.2
const pessoaTensorNormalizada = [
  [
    0.2, // idade normalizada
    0, // cor azul
    0, // cor vermelo
    1, // cor verde
    1, // localização São Paulo
    0, // localização Rio
    0 // localização Curitiba
  ]
]

const predictions = await predict(model, pessoaTensorNormalizada);
const results = predictions
  .sort((a, b) => b.probabilidade - a.probabilidade)
  .map(prob => `${labelsNomes[prob.index]} (${(prob.probabilidade * 100).toFixed(2)}%)`)
  .join('\n')

console.log(results)

