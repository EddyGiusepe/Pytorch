'''
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Link de estudo --> https://towardsdatascience.com/how-to-build-an-image-classification-app-using-logistic-regression-with-a-neural-network-mindset-1e901c938355
'''
import streamlit as st
import numpy as np
from PIL import Image
import time

import torch
from torchvision import transforms

# Definindo as transformações para imagens novas a serem submetidas ao modelo!
image_size = 100

# Transformando as imagens
redimensionamento_imagem = transforms.Compose([
        transforms.Resize(size=[image_size, image_size]),
        transforms.ToTensor(),
    ])

def predicao(model, test_image):
    '''
    Função para realizar a predição do status do AR
    Parâmetros
        :param model: modelo para testar
        :param test_image_name: imagem teste
    '''
    transform = redimensionamento_imagem

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)

    # Não precisa atualizar os coeficientes do modelo
    with torch.no_grad():
        model.eval()

        # Modelo retorna as probabilidades em log (log softmax)
        out = model(test_image_tensor)

        # torch.exp para voltar a probabilidade de log para a probabilidade linear
        ps = torch.exp(out)

        # topk retorna o os k maiores valores do tensor
        # o tensor de probabilidades vai trazer na 1a posição a classe com maior
        # probabilidade de predição
        topk, topclass = ps.topk(3, dim=1)



        classe_com_maior_prob = np.argmax(topk.cpu().numpy()[0])

    return topclass[0][0]


# Designing the interface
st.title("Classificação de bebidas com Pytorch")
# For newline
st.write('\n')

image = Image.open('./images/deep-learning.jpg')
show = st.image(image, use_column_width=True)

st.sidebar.title("Suba uma imagem de bebida!")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Imagem enviada', use_column_width=True)
    # We preprocess the image to fit in algorithm.

# For newline
st.sidebar.write('\n')

# Carregar o modelo
modelo = torch.load('./modelos/melhor_modelo.pt')

if st.sidebar.button("Clique aqui para saber o tipo de bebida."):
    if uploaded_file is None:

        st.sidebar.write("Suba uma imagem para o classificador de bebidas!")

    else:

        with st.spinner('Fazendo a previsão da bebida!!!'):

            prediction = predicao(modelo, u_img)
            time.sleep(2)
            st.success('Pronto!')

        st.sidebar.header("O classificador prevee que é uma ...")

        print(prediction)

        if prediction == 0:
            st.sidebar.write("bebida de chocolate!!!", '\n')
            show.image('./images/previsao_correta.jpg', 'Eu gosta de chocolate!!!', use_column_width=True)
        elif prediction == 1:
            st.sidebar.write("bebida de coca cola!", '\n')
            show.image('./images/nao_cocacola.jpg', 'Não gosto de coca cola!', use_column_width=True)
        elif prediction == 2:
            st.sidebar.write("bebida guarana!", '\n')
            show.image('./images/nao_cocacola.jpg', 'Eu não gosto de Guarana!', use_column_width=True)
