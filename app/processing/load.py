import torch

# Paso 1: Define la arquitectura del modelo (Este código depende de cómo definiste tu modelo)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Inicializa las capas de tu modelo aquí

    def forward(self, x):
        # Define cómo pasa la entrada a través de las capas aquí
        return x

# Paso 2: Carga el checkpoint
checkpoint_path = 'ruta/a/tu/Checkpoint_1706525193928_PCQuorum-SmSmX01_2.checkpoint'
model = MyModel()  # Asegúrate de inicializar tu modelo antes de cargar el estado

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Usa 'cuda' si estás en GPU
model.load_state_dict(checkpoint['model_state_dict'])

# Si el checkpoint incluye también el estado del optimizador y deseas cargarlo
# optimizer = torch.optim.Adam(model.parameters())  # Asegúrate de usar el mismo optimizador y parámetros
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()  # Pon el modelo en modo de evaluación si vas a hacer inferencia
