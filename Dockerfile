# Usar una imagen base oficial, Python 3.11
FROM python:3.10

# Establecer el directorio de trabajo en el contenedor
WORKDIR /./

# Copiar los archivos de requisitos primero para aprovechar la caché de capas de Docker
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Instalar Jupyter
RUN pip install jupyter

# Copiar el resto del codigo fuente al contenedor
COPY . .

# Expone el puerto 8888, el puerto por defecto de Jupyter Notebook
EXPOSE 8888

# Comando para iniciar Jupyter Notebook
# --NotebookApp.token='' desactivar la necesidad de un token para acceder al Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
