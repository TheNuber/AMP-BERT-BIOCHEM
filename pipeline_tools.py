from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from time import process_time_ns 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pandas import DataFrame
from itertools import product
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from time import process_time_ns
import torch


# Define un Dataset y un Dataloader para preprocesar los ejemplos y formar minibatches con ellos, respectivamente

class AMP_Dataset(Dataset):
    """
        Esta clase permite formar un Dataset legible para los modelos de PyTorch
        Implementa los métodos necesarios para entrenar un BERT
    """
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=200):
        super(Dataset, AMP_Dataset).__init__(self)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.df = df
        self.max_len = max_len
        
        self.seqs = list(df['aa_seq'])
        self.labels = list(df['AMP'].astype(int))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_enc = self.tokenizer(
            seq, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors = 'pt',
            return_attention_mask=True
        )
        seq_label = self.labels[idx]
        
        return {
            'idx': idx,
            'input_ids': seq_enc['input_ids'].flatten(),
            'attention_mask' : seq_enc['attention_mask'].flatten(),
            'labels' : torch.tensor(seq_label, dtype=torch.long)
        }
    

class AMP_DataLoader(DataLoader):
    """
        Es una estructura de datos iterable con mini-batches de datos
    
        dataframe   --  Un dataframe de Pandas con los datos, con columnas 'aa_seq' y 'AMP'
        batch_size  --  El tamaño de mini-batch con el que vas a entrenar el modelo   
    """
    def __init__(self, dataframe, batch_size):
        DataLoader.__init__(
            self,
            AMP_Dataset(dataframe),
            batch_size = batch_size,
            num_workers = 2,
            shuffle = True
        )
        
        
# Funcion auxiliar para el calculo del loss
def compute_loss(loss_fn, outputs, labels):
    """ 
        Calcula el loss para realizar el backpropagation
    
        loss_fn -- Funcion de perdida
        outputs -- Los outputs del modelo tal cual        
        labels  -- Etiquetas reales
    """

    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        # La CrossEntropy necesita todos los logits
        return loss_fn(outputs.logits, labels)
    elif isinstance(loss_fn, torch.nn.MSELoss):
        # El MSE necesita el softmax sobre la clase principal (el 1)
        preds = torch.softmax(outputs.logits, dim=1)[:,1:]
        return loss_fn(preds, labels.float())
    else:
        return None

    
def train_model(model, data_loader, loss_fn, optimizer, scheduler, verbose = False):
    """
        Entrena un modelo, y devuelve etiquetas reales, predicciones y el loss final
        
        model         -- El modelo a entrenar
        data_loader   -- un dataloader con los ejemplos de entrenamiento
        loss_fn       -- La funcion de loss (MSE, CrossEntropy, etc.)
        optimizer     -- El optimizador del modelo
        scheduler     -- El scheduler del learning rate del optimizador
        verbose       -- True para mostrar informacion del entrenamiento por consola
    """
    
    model = model.train() # Explicitly setting model to train state
    labels = []
    predictions = []
    losses = []
    correct_predictions = 0
    
    # Variables para calcular una media del loss (no afecta al entrenamiento)
    mobile_loss = 0
    MOBILE_COEF = 0.9
    
    i = 0
    for d in data_loader:
        # Medimos el tiempo
        i = i + 1
        start = process_time_ns()

        # Usamos el batch como input para el modelo y obtenemos el output
        outputs = model(d)

        # Obtenemos las etiquetas del batch
        targets = d['labels'].to("cuda:0")

        # La predicción es la clase con mayor logit
        preds = torch.argmax(outputs.logits, dim = 1)
                
        # Guardamos la prediccion y la etiqueta real, para luego calcular metricas
        labels += targets.tolist()
        predictions += preds.tolist()
                
        # Calculamos el loss
        loss = compute_loss(loss_fn, outputs, targets)
        losses.append(loss.item())
        
        # Calculamos la media movil del loss
        mobile_loss = MOBILE_COEF*mobile_loss + (1-MOBILE_COEF)*loss.item()
        
        # Hacemos el backprop
        loss.backward()
        
        # Clip the gradients of the model to prevent exploding gradients using clip_grad_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Medimos de nuevo
        end = process_time_ns()
        step_time = (end - start) // (10 ** 6)
        remaining_min = (step_time*(len(data_loader) - i) // (10 ** 3)) // 60
        remaining_sec = (step_time*(len(data_loader) - i) // (10 ** 3)) - remaining_min * 60

        # Imprimimos si es necesario
        if verbose:
            if i % 10 == 0:
                print(f"Step {i}/{len(data_loader)}: Loss (avg) {mobile_loss}, Step Time {step_time} ms, ETA {remaining_min}:{remaining_sec}")

    return labels, predictions, losses


def eval_model(model, data_loader, loss_fn, verbose = False):
    """
        Evalua un modelo con un conjunto de datos de test
        
        model         -- El modelo a entrenar
        data_loader   -- un dataloader con los ejemplos de entrenamiento
        loss_fn       -- La funcion de loss (MSE, CrossEntropy, etc.)
        verbose       -- True para mostrar informacion del entrenamiento por consola
    """
    model = model.eval()
    labels = []
    predictions = []
    
    # Variables para calcular una media del loss (no afecta al entrenamiento)
    mobile_loss = 0
    MOBILE_COEF = 0.9

    with torch.no_grad():
        i = 0
        for d in data_loader:
            # Medimos el tiempo
            i = i + 1
            start = process_time_ns()

            # Usamos el batch como input para el modelo y obtenemos el output
            outputs = model(d)

            # Obtenemos las etiquetas del batch
            targets = d['labels'].to("cuda:0")

            # La predicción es la clase con mayor logit
            preds = torch.argmax(outputs.logits, dim = 1)
                
            # Guardamos la prediccion y la etiqueta real, para luego calcular metricas
            labels += targets.tolist()
            predictions += preds.tolist()
            
            # Calculamos el loss
            loss = compute_loss(loss_fn, outputs, targets)
            
            # Calculamos la media movil del loss
            mobile_loss = MOBILE_COEF*mobile_loss + (1-MOBILE_COEF)*loss.item()
            
            # Medimos de nuevo
            end = process_time_ns()
            step_time = (end - start) // (10 ** 6)
            remaining_min = (step_time*(len(data_loader) - i) // (10 ** 3)) // 60
            remaining_sec = (step_time*(len(data_loader) - i) // (10 ** 3)) - remaining_min * 60
    
            # Imprimimos si es necesario
            if verbose:
                if i % 10 == 0:
                    print(f"Step {i}/{len(data_loader)}: Loss (avg) {mobile_loss}, Step Time {step_time} ms, ETA {remaining_min}:{remaining_sec}")

    return labels, predictions


# Define una funcion para calcular métricas sobre etiquetas y predicciones

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from pandas import DataFrame

# Define una funcion para calcular métricas sobre etiquetas y predicciones

def compute_metrics(labels, preds):
    """ 
        Calcula métricas para evaluar el modelo tras un entrenamiento
    
        labels  -- Etiquetas reales
        preds   -- Etiquetas predichas (no logits ni valores no categóricos)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds)
    tn, fp, fn, tp = conf.ravel()
    auroc = roc_auc_score(labels, preds)
    
    precision_data, recall_data, thresholds_data = precision_recall_curve(labels, preds)
    aupr = auc(recall_data, precision_data)
    
    measures = {
        'accuracy': [acc],
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'specificity': [tn / (tn+fp)],
        'auroc': [auroc],
        'aupr': [aupr],
        'confusion_matrix': [conf]
    }
    return DataFrame(measures)



def crossvalidate(model, train_dataframe, n_splits, n_repeats, model_params, loss_fn=torch.nn.CrossEntropyLoss(), verbose=True):
    """
        Crossvalida un modelo
        
        model           -- El modelo a crosvalidar
        train_dataframe -- un dataframe con los ejemplos de entrenamiento (y validacion)
        n_splits        -- Numero de particiones
        n_repeats       -- Numero de repeticiones de la crosvalidacion
        model_params    -- Diccionario con hiperparametros relevantes
        loss_fn         -- La funcion de loss (MSE, CrossEntropy, etc.)
        verbose         -- True para mostrar informacion del entrenamiento por consola
    """
    
    SEED = 0
    BATCH_SIZE = model_params["batch_size"]
    EPOCHS = model_params["epochs"]
    LR = model_params["learning_rate"]
    WARMUP_STEPS = model_params["warmup_steps"]
    WEIGHT_DECAY = model_params["weight_decay"]

    kf = StratifiedKFold(
        n_splits = n_splits,
        shuffle = True,
        random_state = SEED
    )
    
    best_metrics = pd.DataFrame({
        'f1': [0.0]
    })
    
    for i in range(n_repeats):
        if verbose:
            print(f"Crossvalidation: {i+1} repeat")
            print()
        
        # Separamos las secuencias de aminoacidos de la etiqueta para dividirlos
        seqs = train_dataframe['aa_seq']
        labels = train_dataframe['AMP']
        
        val_metrics = pd.DataFrame()
        
        j = 0
        # En esta repeticion entrenamos y evaluamos sobre cada division en train/validation
        for train_j, val_j in kf.split(seqs, labels):
            if verbose:
                j += 1
                print(f"Crossvalidation: {j} fold")
                print()
                
            # Reconstruir los dataframes
            train_df_j = pd.DataFrame({
                'aa_seq': seqs[train_j],
                'AMP': labels[train_j]
            }).sample(frac=1, random_state=SEED)
                        
            val_df_j = pd.DataFrame({
                'aa_seq': seqs[val_j],
                'AMP': labels[val_j]
            }).sample(frac=1, random_state=SEED)
            
            if verbose:
                train_dataloader = AMP_DataLoader(train_df_j, batch_size = BATCH_SIZE)
                print("Sample of train data for this split:")
                print(train_df_j.head(15))
                print()
            
                val_dataloader = AMP_DataLoader(val_df_j, batch_size = BATCH_SIZE)
                print("Sample of validation data for this split:")
                print(val_df_j.head(15))
                print()
            
            # Copiar el modelo para entrenarlo
            model_i = deepcopy(model)
            
            # Entrenar el modelo con esta configuracion
            optimizer_i = AdamW(model_i.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
            
            total_steps_i = len(train_dataloader) * EPOCHS
            
            scheduler_i = get_linear_schedule_with_warmup(optimizer_i, 
                                           num_warmup_steps = WARMUP_STEPS, 
                                           num_training_steps = total_steps_i)

            for i in range(EPOCHS):
                train_model(model_i, train_dataloader, loss_fn, optimizer_i, scheduler_i, verbose=False)

            # Obtener las métricas de validacion
            val_labels, val_preds = eval_model(model_i, val_dataloader, loss_fn, verbose=False)
            
            val_metrics_i = compute_metrics(val_labels, val_preds)
            
            if verbose:
                print(f"Metrics for fold {j}: ")
                print(val_metrics_i)
                print()
            
            val_metrics = pd.concat([val_metrics, val_metrics_i], ignore_index=True)
            
        # Una vez probado con cada division de train/validation en este set, calculamos la media de las metricas
        mean_metrics = val_metrics.mean(axis=0)
        if verbose:
            print(f"Mean metrics for repeat {i+1}: ")
            print(mean_metrics)
            print()
        
        if mean_metrics['f1'].item() >= best_metrics['f1'].item():
            best_metrics = mean_metrics
    
    return best_metrics

from torch.nn import CrossEntropyLoss

def grid_search_bert_model(model, train_val_dataframe, grid, loss_fn = CrossEntropyLoss(), verbose = False):
        
    param_combinations = product(
        grid["epochs"],
        grid["batch_size"],
        grid["learning_rate"],
        grid["betas"],
        grid["epsilon"],
        grid["weight_decay"],
        grid["warmup_steps"]
    )
    
    all_combs = []
    all_metrics = []
    all_losses = []
        
    # Calculamos todas las combinaciones con el grid de hiperparametros
    num_combinations = 1
    for key in grid.keys():
        num_combinations *= len(grid[key])
    
    # Separamos en train y validacion
    TRAIN_FRAC = 0.8
    
    train_dataframe = train_val_dataframe.sample(frac=TRAIN_FRAC, random_state=SEED)
    val_dataframe = train_val_dataframe.drop(train_dataframe.index)
    
    print()
    print(f"Number of combinations: {num_combinations}")

    for combination in param_combinations:
        
        # En cada combinacion entrenamos y testeamos
        epochs, batch_size, learning_rate, betas, epsilon, weight_decay, warmup_steps = combination
        
        train_data_loader = AMP_DataLoader(train_dataframe, batch_size = batch_size)
        val_data_loader = AMP_DataLoader(val_dataframe, batch_size = batch_size)

        print()
        print("Next combination:")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"learning_rate: {learning_rate}")
        print(f"betas: {betas}")
        print(f"epsilon: {epsilon}")
        print(f"weight_decay: {weight_decay}")
        print(f"warmup_steps: {warmup_steps}")
        
        # Copiamos el modelo
        model_copy = deepcopy(model)
        
        # Preparamos el optimizador y el scheduler
        optimizer = AdamW(
            model_copy.parameters(), 
            lr = learning_rate,
            betas = betas,
            eps = epsilon,
            weight_decay = weight_decay
        )
        
        total_steps = len(train_data_loader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps = warmup_steps, 
                                           num_training_steps = total_steps)
        # Entrenamos
        train_start = process_time_ns()
        for epoch in range(epochs):
            _, _, losses = train_model(model_copy, train_data_loader, loss_fn, optimizer, scheduler, verbose)
        train_end = process_time_ns()
        
        # Medimos
        eval_start = process_time_ns()
        labels, predictions = eval_model(model_copy, val_data_loader, loss_fn, verbose)
        eval_end = process_time_ns()

        metrics = compute_metrics(labels, predictions)
        metrics["train_time_secs"] = (train_end - train_start) // (10 ** 9)
        metrics["eval_time_secs"] = (eval_end - eval_start) // (10 ** 9)
        
        # Guardamos las medidas
        all_combs.append(combination)
        all_metrics.append(metrics)
        all_losses.append(losses)
                    
        del model_copy
        
    df_combs = pd.DataFrame(all_combs, index = range(num_combinations), columns=['epochs', 'batch_size', 'learning_rate', 'betas', 'epsilon', 'weight_decay', 'warmup_steps'])
    df_metrics = pd.concat(all_metrics)
    df_metrics.index = range(num_combinations)
    df_results = pd.concat([df_combs, df_metrics], axis=1)
    
    df_losses = pd.DataFrame(all_losses, index = range(num_combinations))
    
    return df_results, df_losses


def get_embeddings(model, data_loader):
    """
        Obtiene los embeddings del modelo asociados a los datos
        
        model         -- El modelo a usar
        data_loader   -- un dataloader con los ejemplos de entrenamiento
    """
    model = model.eval()
    
    indexes = []
    labels = []
    predictions = []
    embeddings = torch.tensor([]).to("cuda:0")
    
    with torch.no_grad():
        for d in data_loader:
            # Obtenemos los atributos del siguiente batch
            idx = d['idx']
            input_ids = d['input_ids'].to("cuda:0")
            attention_mask = d['attention_mask'].to("cuda:0")
            targets = d['labels'].to("cuda:0")
        
            # Lo usamos como input para el modelo y obtenemos el output
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            preds = torch.argmax(outputs.logits, dim = 1)

            indexes.append(idx)
            labels += targets.tolist()
            predictions += preds.tolist()
            embeddings = torch.cat((embeddings, outputs.hidden_states), dim = 0)
            
    return indexes, labels, predictions, embeddings

               
class AMP_BioChemDataset(Dataset):
    """
        Esta clase permite formar un Dataset legible para los modelos de PyTorch
        Implementa los métodos necesarios para entrenar un BERT
    """
    def __init__(self, df, biochem_cols, tokenizer_name='Rostlab/prot_bert_bfd', max_len=200):
        super(Dataset, AMP_BioChemDataset).__init__(self)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.df = df
        self.max_len = max_len
        
        self.seqs = list(df['aa_seq'])
        self.biochem_cols = biochem_cols
        self.biochem = df[biochem_cols]
        self.labels = list(df['AMP'].astype(int))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_enc = self.tokenizer(
            seq, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors = 'pt',
            return_attention_mask=True
        )
        seq_label = self.labels[idx]
        seq_biochem = self.biochem.iloc[idx]
        if "molecular_mass" in self.biochem_cols:
            seq_biochem.loc['molecular_mass'] = seq_biochem['molecular_mass'] / 1e4
        seq_biochem.transpose()
        
        return {
            'idx': idx,
            'input_ids': seq_enc['input_ids'].flatten(),
            'attention_mask' : seq_enc['attention_mask'].flatten(),
            'labels' : torch.tensor(seq_label, dtype=torch.long),
            'biochem_info': torch.tensor(seq_biochem, dtype=torch.float32)
        }
    

class AMP_BioChemDataLoader(DataLoader):
    """
        Es una estructura de datos iterable con mini-batches de datos
    
        dataframe   --  Un dataframe de Pandas con los datos, con columnas 'aa_seq' y 'AMP'
        batch_size  --  El tamaño de mini-batch con el que vas a entrenar el modelo   
    """
    def __init__(self, dataframe, biochem_cols, batch_size):
        DataLoader.__init__(
            self,
            AMP_BioChemDataset(dataframe, biochem_cols),
            batch_size = batch_size,
            num_workers = 2,
            shuffle = True
        )
        