import torch

def averageModels(global_model, clients):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    samples = [clients[i]['samples'] for i in range(len(clients))]
    global_dict = global_model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))], 0).sum(0)
            
    global_model.load_state_dict(global_dict)
    return global_model


def averageGradients(global_model, clients):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    samples = [clients[i]['samples'] for i in range(len(clients))]

    for k in range(len(list(client_models[0].parameters()))):
        list(global_model.parameters())[k].grad = torch.stack([list(client_models[i].parameters())[k].grad.clone() * samples[i] for i in range(len(client_models))], 0).sum(0)
    return global_model